English | [中文](README_zh.md)

# PPO End-to-End Navigation: INT8 Quantization × Action Chunking (N=5)

> Combines **INT8 quantization** and **action chunking (N=5)** for end-to-end 2D LiDAR robot navigation.
> Core deliverables in this repository: training/evaluation pipeline (`src/`, `env/`, `configs/`), ONNX export scripts (`onnx/`), the `deploy/ppo_nav/` ROS1 deployment package, and the `runs/ablation_runbook.md` on-robot protocol.

---

## Project Positioning

- **Problem being solved**: End-to-end RL navigation inference latency and control overhead on embedded hardware. Two orthogonal levers address inference cost and execution behavior:
  - **INT8 quantization**: ONNX → TensorRT PTQ; Jetson Orin Nano (`TRT 8.5`) batch=1 inference latency **-33.0%** vs FP32 on the equivalent `1024×1024` pure-GEMM Actor ONNX (measured in `runs/trt_bench_results.md`);
  - **action chunking (N=5)**: one inference emits 5 consecutive steps executed open-loop, cutting inference frequency to 1/5 and improving **trajectory smoothness / temporal consistency** across adjacent control steps.
- **Main experiment**: navigation with **N=5 + INT8 quantization both active** (compared against N=5/FP32, N=1/INT8, and N=1/FP32 in the 4-cell on-robot ablation matrix).
- **Policy architecture**: SB3 `PPO` (`MlpPolicy`, `1024×1024`, `Tanh`, pure GEMM/Tanh graph).

---

## Implemented Pipeline & Archived Records

> Note: Trained model weights (`checkpoints/`), exported ONNX/calibration files (`export/`), and local training logs (`runs/ppo_*`) are gitignored and **not bundled in this repository**; generate them locally by following the Quick Start below.

| Item | Status / Scope |
| --- | --- |
| PPO training (`N=1` / `N=5`, `ir-sim`, `83ms` step) | Supported via `src/train_ppo.py --chunk {1,5}` (outputs to local `checkpoints/ppo_final_N{1,5}/`) |
| PPO simulation evaluation (success / collision / timeout) | Supported via `src/eval_ppo.py --checkpoint ...` (auto-detects N=1 or N=5) |
| PPO → ONNX export (pure `Gemm`/`Tanh`, opset 17) | Supported via `onnx/unpack_ppo_actor.py` + `onnx/export_ppo_onnx.py` (N=1 → 105 dim, N=5 → 113 dim) |
| INT8 throughput & precision benchmark (PC / Orin) | Recorded in `runs/trt_bench_results.md`: Orin/TRT 8.5 bs1 INT8 **-33.0%** vs FP32 (on the equivalent `1024×1024` pure-GEMM Actor ONNX) |
| INT8 vs FP32 collision comparison (sim, N=1 only) | Recorded in `runs/quant_collision/quant_collision_results.md` via `src/eval_quant_collision.py`: 300 paired episodes on earlier N=1 `ppo_mw_N1` (PyTorch fake-quant, `0.3s` step, `max_steps=500`), max-abs diff `0.00875`, collision rate `+1.00%` (note: N=5 has no separate `eval_quant_collision` script) |

---

## Environment Requirements

**PC (training / evaluation / ONNX export — single unified conda environment)**

- Python >= 3.10
- Create one conda environment and install dependencies from `requirements.txt`:

```bash
conda create -n ppo_nav python=3.10 -y
conda activate ppo_nav
pip install -r requirements.txt
```

**Robot (deployment)**

- ROS1 Noetic + Jetson Orin Nano (JetPack R35.6.1, TensorRT 8.5, CUDA)
- Deployment package: `deploy/ppo_nav/` (see `deploy/README.md`)

---

## Directory Structure

```text
PPO-End-to-End-Navigation-with-model-quantilize-and-action-chunk/
├─ configs/             # train.yaml (master training/eval config) + world/ scenarios (ir-sim YAML)
├─ env/                 # ir-sim env wrappers: wrapper.py (IrSimEnv), reward.py, ppo_gym.py, multi_world.py
├─ src/                 # train_ppo.py (N=1/N=5), eval_ppo.py (N=1/N=5), eval_quant_collision.py (N=1 sim fake-quant)
├─ onnx/                # unpack_ppo_actor.py / export_ppo_onnx.py (two-step ONNX export)
├─ deploy/              # real-robot deployment: ppo_nav ROS1 package + build_ptq_engine.py + README
├─ doc/                 # PPO network structure analysis (md + reproduction script + archived report)
├─ runs/                # ablation_runbook.md + trt_bench_results.md + quant_collision/
├─ utils/               # config.py: config loading / deep merge / path resolution
├─ checkpoints/         # (generated locally by training, gitignored)
└─ export/              # (generated locally by ONNX export, gitignored)
```

---

## Quick Start: Train → Export → Quantize → Deploy

### 1) Training (N=1 and N=5)

```bash
# Full training (GPU, multi-world; chunk=1 outputs checkpoints/ppo_final_N1/, chunk=5 outputs checkpoints/ppo_final_N5/)
python src/train_ppo.py --steps 300000 --device cuda --name ppo_final --chunk 1 --ent-coef 0.01 --n-steps 4096
python src/train_ppo.py --steps 300000 --device cuda --name ppo_final --chunk 5 --ent-coef 0.01 --n-steps 4096
```

> Entropy-collapse guard: at the `83ms` (`0.0833s`) step size use `--n-steps 4096 --ent-coef 0.01`; export from `best_model.zip` (saved by eval success rate), not `ppo_model.zip`.

### 2) Evaluation (optional, simulation validation)

```bash
python src/eval_ppo.py --checkpoint checkpoints/ppo_final_N1/best_model.zip
python src/eval_ppo.py --checkpoint checkpoints/ppo_final_N5/best_model.zip
```

### 3) Export ONNX + collect calibration data (in the same conda env)

```bash
# ① Unpack Actor → export/ppo_final_N1/, export/ppo_final_N5/
python onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N1/best_model.zip --name ppo_final_N1
python onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N5/best_model.zip --name ppo_final_N5

# ② Export ONNX + INT8 calibration data
python onnx/export_ppo_onnx.py --actor export/ppo_final_N1/policy_actor.pt --make-calib
python onnx/export_ppo_onnx.py --actor export/ppo_final_N5/policy_actor.pt --make-calib --chunk 5
```

Generated artifacts (`export/ppo_final_N{1,5}/`): `actor_fp32_bs1.onnx`, `actor_fp32_bs8.onnx` (pure `Gemm`/`Tanh`), `calib_obs.npz` (real-obs calibration set), `policy_actor.pt`, `actor_config.yaml`.

> Note: `onnx/export_ablation.ps1` still references legacy `train\` paths and is kept untouched; run the commands above directly.

### 4) Build engines on board (FP32 baseline + INT8 quantization; batch=1)

```bash
scp -r export/ppo_final_N1 export/ppo_final_N5 wheeltec@<ip>:~/ppo_deploy/
cd ~/ppo_deploy/ppo_final_N1
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32_n1.engine
python3 ~/wheeltec_robot/src/ppo_nav/scripts/build_ptq_engine.py \
    --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8_n1.engine --batch-size 1

cd ../ppo_final_N5
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32_n5.engine
python3 ~/wheeltec_robot/src/ppo_nav/scripts/build_ptq_engine.py \
    --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8_n5.engine --batch-size 1
```

### 5) Real-robot deployment and benchmarking

```bash
scp -r deploy/ppo_nav/ wheeltec@<ip>:~/wheeltec_robot/src/
# Build + copy the 4 engines to models/ + launch via planner.launch args; full procedure in runs/ablation_runbook.md
roslaunch ppo_nav planner.launch
```

**On-robot 4-cell ablation matrix** (`runs/ablation_runbook.md`):

| Cell | Quantization | chunk | Engine | `chunk_size` | `use_prev_action` | obs dim |
|---|---|---|---|---|---|---|
| ① | FP32 | N=5 | `actor_fp32_n5.engine` | 5 | true | 113 |
| ② | **INT8** | **N=5 (main experiment)** | `actor_int8_n5.engine` | 5 | true | 113 |
| ③ | INT8 | N=1 | `actor_int8_n1.engine` | 1 | false | 105 |
| ④ | FP32 | N=1 | `actor_fp32_n1.engine` | 1 | false | 105 |

---

## Observation / Action / Reward Contract (`IrSimEnv` in `env/wrapper.py`)

- **Observation**:
  - `lidar`: binned-min then divided by `obs.lidar_range_max` (`obs.max_bins`, default `100` = lidar beam count)
  - `goal` polar coordinates: `[dist / goal_dist_norm, cos, sin]` (3 dims, `goal_dist_norm` default `10.0`)
  - `previous action` (`obs.include_prev_action: true`): `N * action_dim` dims, per-step `[lin*2, (ang+1)/2]`
    - N=1: trained through `step_single`, this channel is always 0 (deployment `use_prev_action=false`)
    - N=5: during chunk training writes back the previous chunk's 5 actions (deployment `use_prev_action=true`, planner accumulates `prev_history_`)
- **Action**: normalized `[-1, 1]`, mapped to real velocities (`[linear_vel, angular_vel]`); PPO output is μ, clipped to `[-1, 1]` before scaling.
- **Reward** (`configs/train.yaml` + `env/reward.py`):
  - Arrival: `+400.0` (`goal_reward` in `configs/train.yaml`; `RewardFn` fallback default `100.0`)
  - Collision: `-100.0` (`collision_penalty`)
  - Non-terminal step: `lin - 0.5|ang| - 0.1 (time_penalty) - proximity_penalty + goal_shaping` (`0.5 * cos(angle_to_goal) + 1.0 * (prev_dist - dist)`)
- **Termination**: arrival / collision / timeout (`max_steps: 1800` at `0.0833s` step).

## Replacing the Environment

Place `ir-sim` world YAML files in `configs/world/`, then configure `configs/train.yaml`:
- `env.worlds` (non-empty by default) **takes precedence over `env.world_name`** in `src/train_ppo.py`. To train on a single custom world, either set `env.worlds: []` and update `env.world_name`, or update the entries in `env.worlds`.
- Update `env.eval_world` for the evaluation scenario.
- World requirements: robot equipped with a `lidar2d` sensor whose beam count matches `obs.max_bins` / `obs.lidar_len` (`100`) and whose `range_max` matches `obs.lidar_range_max` (`7.0`), plus a `goal`.
