[English](ppo_structure_analysis.md) | 中文

# PPO 模型结构与契约说明（N=1 & N=5）

> 用途：说明本仓库 **N=1**（`ppo_final_N1`）与 **N=5**（`ppo_final_N5`）两类 SB3 PPO 策略的网络结构、精确参数量以及量化/部署契约。
> 仓库未内置训练权重（`checkpoints/` 已 gitignore）。本地训练完成后可运行 `python doc/analyze_ppo_structure.py` 检查实际 `.zip` 权重；`doc/ppo_structure_report.txt` 为早期 N=1 模型（`ppo_N1` / `ppo_smoke_N1`）的归档打印样例。

---

## 1. 策略网络结构总览

N=1 与 N=5 均采用相同的 SB3 `PPO`（`MlpPolicy` / `ActorCriticPolicy`）配置（`net_arch=[1024, 1024]`）：

- 算法：Stable-Baselines3 `PPO`，`ActorCriticPolicy`，连续对角高斯分布（`DiagGaussianDistribution`）
- 激活函数：**Tanh**（仅隐藏层）
- `squash_output=False`：输出层无 `tanh` 压缩，确定性动作为 `clip(μ, [-1, 1])`
- `share_features_extractor=True`：对 `MlpPolicy` 仅相当于恒等 `FlattenExtractor`，`mlp_extractor.policy_net` 与 `mlp_extractor.value_net` 为**两个独立 MLP，不共享权重**

### 1.1 网络拓扑与参数量

#### 变体 A：N=1 单步模型（`obs_dim=105`，`action_dim=2`）

```text
obs (105,) ──┬─> Actor（导出 ONNX 部分）：               参数量        MACs
             │    Linear(105 -> 1024) + Tanh          108,544     107,520
             │    Linear(1024 -> 1024) + Tanh       1,049,600   1,048,576
             │    Linear(1024 -> 2)                     2,050       2,048
             │    └─ μ (2,)  [SB3 中另有 2 个可学习 log_std]
             └─> Critic（仅训练期使用）：
                  Linear(105 -> 1024) + Tanh          108,544
                  Linear(1024 -> 1024) + Tanh       1,049,600
                  Linear(1024 -> 1)                     1,025
                  └─ V (1,)
```

- Actor 参数量（不含 `log_std`）：**1,160,194**（约 1.16M）
- SB3 `policy` 总参数量（Actor + `log_std` + Critic）：**2,319,365**（约 2.32M）
- Actor 单次前向 MACs：**1,158,144**（约 1.16M）

#### 变体 B：N=5 Action Chunk 模型（`obs_dim=113`，`action_dim=10`）

```text
obs (113,) ──┬─> Actor（导出 ONNX 部分）：               参数量        MACs
             │    Linear(113 -> 1024) + Tanh          116,736     115,712
             │    Linear(1024 -> 1024) + Tanh       1,049,600   1,048,576
             │    Linear(1024 -> 10)                   10,250      10,240
             │    └─ μ (10,) = 5×[lin, ang] [SB3 中另有 10 个 log_std]
             └─> Critic（仅训练期使用）：
                  Linear(113 -> 1024) + Tanh          116,736
                  Linear(1024 -> 1024) + Tanh       1,049,600
                  Linear(1024 -> 1)                     1,025
                  └─ V (1,)
```

- Actor 参数量（不含 `log_std`）：**1,176,586**（约 1.18M）
- SB3 `policy` 总参数量（Actor + `log_std` + Critic）：**2,343,957**（约 2.34M）
- Actor 单次前向 MACs：**1,174,528**（约 1.175M）

---

## 2. 训练超参数（`src/train_ppo.py` + `configs/train.yaml`）

| 超参数 | 推荐主训练值（`ppo_final`，83ms 步长） | `train_ppo.py` CLI 缺省值 |
| --- | --- | --- |
| `step_time` | `0.0833s`（12Hz，配置在 `configs/train.yaml`） | 读取 `configs/train.yaml` |
| `max_steps` | `1800`（配置在 `configs/train.yaml`） | 读取 `configs/train.yaml` |
| `n_steps` | `4096`（`--n-steps 4096`） | `1024` |
| `ent_coef` | `0.01`（`--ent-coef 0.01`） | `0.0` |
| `learning_rate` | `3e-4` | `3e-4` |
| `batch_size` | `128` | `128` |
| `n_epochs` | `10` | `10` |
| `gamma` | `0.99` | `0.99` |
| `gae_lambda` | `0.95` | `0.95` |
| `clip_range` | `0.2`（常数） | `0.2` |
| `vf_coef` | `0.5` | `0.5` |
| `max_grad_norm` | `0.5` | `0.5` |

---

## 3. 对 INT8 量化与 Action Chunking 的影响

### 3.1 INT8 量化

- 部署只需提取 **Actor** 子网（`mlp_extractor.policy_net` + `action_net`）导出为 ONNX（`[B, obs_dim] -> [B, action_dim]`，FP32）。
- 导出图仅含 **`Gemm` + `Tanh`**（opset 17），**无 `Conv`、无 `BatchNorm`**（观测归一化在 `IrSimEnv` / `ObsBuilder` 完成），TensorRT 可全图降精度执行，避免 Reformat cast 开销。
- 网络输出 μ 未经激活压缩，部署端需先裁剪至 `[-1, 1]` 再映射速度。
- 导出链路：`onnx/unpack_ppo_actor.py` → `onnx/export_ppo_onnx.py`（`--make-calib`），车端使用 `deploy/ppo_nav/scripts/build_ptq_engine.py`（`--batch-size 1`）构建 INT8 引擎。

### 3.2 Action Chunking（`N=5`）

- 训练端（`src/train_ppo.py --chunk 5`）：`PPOGymEnv` 将动作空间扩展为 `(2*N,) = (10,)`，每次决策调用 `IrSimEnv.step_chunk` 开环执行 5 步，并将执行完的 5 步动作写回观测的“上一动作”通道（`5 * 2 = 10` 维，总 `obs_dim = 113`）。
- 部署端（`deploy/ppo_nav`）：`planner_node` 每 `chunk_size=5` 帧 `/scan` 推理一次，将 5 步动作放入 `pending_`，后续帧逐拍弹出（开环），并在 `use_prev_action=true` 时通过 `ObsBuilder::setPrevChunk` 写回上一 chunk 动作。

### 3.3 N=1 与 N=5 契约对比

| 项目 | N=1（`ppo_final_N1`） | N=5（`ppo_final_N5`） |
| --- | --- | --- |
| `obs` | `105` = `100` 激光 + `3` 目标 + `2` 上一动作（训练期恒为 0） | `113` = `100` 激光 + `3` 目标 + `10` 上一 chunk 动作 |
| `action` | `[-1, 1]²`（单步 `[lin, ang]`） | `[-1, 1]¹⁰`（5 步 × `[lin, ang]`） |
| Actor 参数量（不含 `log_std`） | `1,160,194`（约 1.16M），`1.158M` MACs | `1,176,586`（约 1.18M），`1.175M` MACs |
| 部署 launch 参数 | `chunk_size:=1 use_prev_action:=false` | `chunk_size:=5 use_prev_action:=true` |

---

## 4. 检查本地训练后的权重

在本地训练生成 `checkpoints/` 后，可运行脚本检查实际模型结构：

```bash
python doc/analyze_ppo_structure.py
# 或显式指定要检查的 checkpoint 路径：
python doc/analyze_ppo_structure.py checkpoints/ppo_final_N1/best_model.zip checkpoints/ppo_final_N5/best_model.zip
```
