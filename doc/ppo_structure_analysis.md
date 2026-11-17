English | [中文](ppo_structure_analysis_zh.md)

# PPO Model Structure Specification (N=1 & N=5)

> Purpose: document the SB3 PPO policy architecture, exact parameter counts, and deployment contracts for both **N=1** (`ppo_final_N1`) and **N=5** (`ppo_final_N5`) models in this repository.
> Checkpoints are not pre-bundled in the repo (`checkpoints/` is gitignored). Once trained locally, you can inspect any `.zip` with `python doc/analyze_ppo_structure.py`. An archived raw dump from earlier N=1 checkpoints (`ppo_N1` / `ppo_smoke_N1`) is kept at `doc/ppo_structure_report.txt` for reference.

---

## 1. Policy Architecture Overview

Both N=1 and N=5 share the same SB3 `PPO` (`MlpPolicy` / `ActorCriticPolicy`) configuration (`net_arch=[1024, 1024]`):

- Algorithm: Stable-Baselines3 `PPO`, `ActorCriticPolicy`, continuous diagonal Gaussian (`DiagGaussianDistribution`)
- Activation: **Tanh** (hidden layers only)
- `squash_output=False`: no output `tanh`; deterministic action is `clip(μ, [-1, 1])`
- `share_features_extractor=True`: for `MlpPolicy` this uses an identity `FlattenExtractor`, while `mlp_extractor.policy_net` and `mlp_extractor.value_net` are **two independent MLPs with no shared weights**

### 1.1 Network Topology & Parameter Counts

#### Variant A: N=1 Single-Step (`obs_dim=105`, `action_dim=2`)

```text
obs (105,) ──┬─> Actor (exported to ONNX):            params       MACs
             │    Linear(105 -> 1024) + Tanh         108,544    107,520
             │    Linear(1024 -> 1024) + Tanh      1,049,600  1,048,576
             │    Linear(1024 -> 2)                    2,050      2,048
             │    └─ μ (2,)  [+ 2 learnable log_std params in SB3]
             └─> Critic (training only):
                  Linear(105 -> 1024) + Tanh         108,544
                  Linear(1024 -> 1024) + Tanh      1,049,600
                  Linear(1024 -> 1)                    1,025
                  └─ V (1,)
```

- Actor params (excl. `log_std`): **1,160,194** (~1.16M)
- Total SB3 `policy` params (Actor + `log_std` + Critic): **2,319,365** (~2.32M)
- Actor forward MACs: **1,158,144** (~1.16M)

#### Variant B: N=5 Action Chunk (`obs_dim=113`, `action_dim=10`)

```text
obs (113,) ──┬─> Actor (exported to ONNX):            params       MACs
             │    Linear(113 -> 1024) + Tanh         116,736    115,712
             │    Linear(1024 -> 1024) + Tanh      1,049,600  1,048,576
             │    Linear(1024 -> 10)                  10,250     10,240
             │    └─ μ (10,) = 5×[lin, ang] [+ 10 log_std params in SB3]
             └─> Critic (training only):
                  Linear(113 -> 1024) + Tanh         116,736
                  Linear(1024 -> 1024) + Tanh      1,049,600
                  Linear(1024 -> 1)                    1,025
                  └─ V (1,)
```

- Actor params (excl. `log_std`): **1,176,586** (~1.18M)
- Total SB3 `policy` params (Actor + `log_std` + Critic): **2,343,957** (~2.34M)
- Actor forward MACs: **1,174,528** (~1.175M)

---

## 2. Training Hyperparameters (`src/train_ppo.py` + `configs/train.yaml`)

| Hyperparameter | Recommended (`ppo_final`, 83ms step) | `train_ppo.py` CLI default |
| --- | --- | --- |
| `step_time` | `0.0833s` (12Hz, in `configs/train.yaml`) | from `configs/train.yaml` |
| `max_steps` | `1800` (in `configs/train.yaml`) | from `configs/train.yaml` |
| `n_steps` | `4096` (`--n-steps 4096`) | `1024` |
| `ent_coef` | `0.01` (`--ent-coef 0.01`) | `0.0` |
| `learning_rate` | `3e-4` | `3e-4` |
| `batch_size` | `128` | `128` |
| `n_epochs` | `10` | `10` |
| `gamma` | `0.99` | `0.99` |
| `gae_lambda` | `0.95` | `0.95` |
| `clip_range` | `0.2` (constant) | `0.2` |
| `vf_coef` | `0.5` | `0.5` |
| `max_grad_norm` | `0.5` | `0.5` |

---

## 3. Implications for INT8 Quantization & Action Chunking

### 3.1 INT8 Quantization

- Only the **Actor** subgraph (`mlp_extractor.policy_net` + `action_net`) is unpacked and exported to ONNX (`[B, obs_dim] -> [B, action_dim]`, FP32).
- The exported ONNX contains only **`Gemm` + `Tanh`** (opset 17), with **no `Conv` and no `BatchNorm`** (observation normalization happens in `IrSimEnv` / `ObsBuilder`). TensorRT can execute the full graph without conv/reformat casts.
- Output μ is un-squashed and clipped to `[-1, 1]` on the deployment side before `scale_action`.
- Export pipeline: `onnx/unpack_ppo_actor.py` → `onnx/export_ppo_onnx.py` (`--make-calib`), followed by on-board `deploy/ppo_nav/scripts/build_ptq_engine.py` (`--batch-size 1`).

### 3.2 Action Chunking (`N=5`)

- Training (`src/train_ppo.py --chunk 5`): `PPOGymEnv` expands the action space to `(2*N,) = (10,)` and executes `IrSimEnv.step_chunk` open-loop for 5 steps per decision, writing the executed 5-step chunk back into the previous-action observation channel (`5 * 2 = 10` dims, total `obs_dim = 113`).
- Deployment (`deploy/ppo_nav`): `planner_node` runs inference once every `chunk_size=5` `/scan` frames, pushes the 5 action pairs into `pending_`, pops one pair per frame open-loop, and with `use_prev_action=true` writes the executed chunk back into `ObsBuilder::setPrevChunk`.

### 3.3 Contract Comparison Summary

| Item | N=1 (`ppo_final_N1`) | N=5 (`ppo_final_N5`) |
| --- | --- | --- |
| `obs` | `105` = `100` lidar + `3` goal + `2` prev action (always 0 in training) | `113` = `100` lidar + `3` goal + `10` prev chunk |
| `action` | `[-1, 1]²` (single `[lin, ang]`) | `[-1, 1]¹⁰` (5 steps × `[lin, ang]`) |
| Actor params (excl. `log_std`) | `1,160,194` (~1.16M), `1.158M` MACs | `1,176,586` (~1.18M), `1.175M` MACs |
| Deployment args | `chunk_size:=1 use_prev_action:=false` | `chunk_size:=5 use_prev_action:=true` |

---

## 4. Inspecting Local Checkpoints

After training checkpoints into `checkpoints/`, inspect them with:

```bash
python doc/analyze_ppo_structure.py
# Or specify exact checkpoint paths:
python doc/analyze_ppo_structure.py checkpoints/ppo_final_N1/best_model.zip checkpoints/ppo_final_N5/best_model.zip
```
