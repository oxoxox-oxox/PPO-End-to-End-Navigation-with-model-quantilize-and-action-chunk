[English](README.md) | 中文

# PPO 端到端导航：INT8 量化 × Action Chunking（N=5）

> 将 **INT8 量化** 与 **Action Chunking（N=5）** 结合用于端到端 2D 激光雷达机器人导航。
> 仓库核心内容包括：训练/评估代码（`src/`、`env/`、`configs/`）、ONNX 导出脚本（`onnx/`）、`deploy/ppo_nav/` ROS1 部署包，以及 `runs/ablation_runbook_zh.md` 实车实验手册。

---

## 项目定位

- **解决的问题**：端到端 RL 导航在嵌入式端推理开销与控制响应问题，通过两条正交杠杆协同优化：
  - **INT8 量化**：ONNX → TensorRT PTQ，Jetson Orin Nano（`TRT 8.5`）上针对同结构 `1024×1024` 纯 GEMM Actor ONNX 的 batch=1 推理延迟 **-33.0%**（见 `runs/trt_bench_results_zh.md`）；
  - **Action Chunking（N=5）**：单次推理输出连续 5 步动作并开环执行，将推理频率降至 1/5，并提升相邻控制步间的**轨迹平滑性与时间一致性**。
- **主实验**：**N=5 + INT8 量化同时开启**（与 N=5/FP32、N=1/INT8、N=1/FP32 构成实车 4 格消融对照矩阵）。
- **策略网络**：SB3 `PPO`（`MlpPolicy`，`1024×1024`，`Tanh`，全图仅 `Gemm`/`Tanh`）。

---

## 当前代码能力与实验记录

> 说明：训练权重（`checkpoints/`）、导出的 ONNX 与校准集（`export/`）以及训练日志（`runs/ppo_*`）均被 `.gitignore` 忽略，**未内置于仓库中**；请按下方 Quick Start 在本地训练并导出生成。

| 项目 | 当前状态 / 适用范围 |
| --- | --- |
| PPO 训练（`N=1` / `N=5`，`ir-sim`，`83ms` 步长） | 已实现：`src/train_ppo.py --chunk {1,5}`（输出至本地 `checkpoints/ppo_final_N{1,5}/`） |
| PPO 仿真评估（成功率 / 碰撞率 / 超时率） | 已实现：`src/eval_ppo.py --checkpoint ...`（自动识别 N=1 或 N=5） |
| PPO → ONNX 导出（纯 `Gemm`/`Tanh`，opset 17） | 已实现：`onnx/unpack_ppo_actor.py` + `onnx/export_ppo_onnx.py`（N=1→105 维 / N=5→113 维） |
| INT8 吞吐与精度基准（PC / Orin） | 已记录：`runs/trt_bench_results_zh.md`（Orin/TRT 8.5 bs1 INT8 相对 FP32 **-33.0%**，基于同结构 `1024×1024` 纯 GEMM Actor ONNX） |
| INT8 vs FP32 仿真碰撞对比（仅 N=1） | 已记录：`runs/quant_collision/quant_collision_results_zh.md`（由 `src/eval_quant_collision.py` 运行：基于早期 N=1 模型 `ppo_mw_N1`、PyTorch fake-quant、`0.3s` 步长、`max_steps=500`、300 组配对回合，max-abs `0.00875`，碰撞率 `+1.00%`；注：N=5 暂无单独的 `eval_quant_collision` 脚本） |

---

## 环境要求

**PC 端（训练 / 评估 / ONNX 导出——单个统一 Conda 环境）**

- Python >= 3.10
- 创建单个 conda 环境并安装 `requirements.txt`：

```bash
conda create -n ppo_nav python=3.10 -y
conda activate ppo_nav
pip install -r requirements.txt
```

**车端（实车部署）**

- ROS1 Noetic + Jetson Orin Nano（JetPack R35.6.1，TensorRT 8.5，CUDA）
- 部署包位置：`deploy/ppo_nav/`（详见 `deploy/README_zh.md`）

---

## 目录结构

```text
PPO-End-to-End-Navigation-with-model-quantilize-and-action-chunk/
├─ configs/             # train.yaml（训练/评估主配置）+ world/（ir-sim 场景 YAML）
├─ env/                 # ir-sim 环境封装：wrapper.py (IrSimEnv)、reward.py、ppo_gym.py、multi_world.py
├─ src/                 # train_ppo.py (N=1/N=5)、eval_ppo.py (N=1/N=5)、eval_quant_collision.py (N=1 仿真伪量化对比)
├─ onnx/                # unpack_ppo_actor.py / export_ppo_onnx.py（两步式 ONNX 导出）
├─ deploy/              # 实车部署：ppo_nav ROS1 包 + build_ptq_engine.py + README
├─ doc/                 # PPO 网络结构分析（md + 复现脚本 + 归档报告）
├─ runs/                # ablation_runbook_zh.md + trt_bench_results_zh.md + quant_collision/
├─ utils/               # config.py：配置加载 / 深度合并 / 路径解析
├─ checkpoints/         # （本地训练生成，已 gitignore）
└─ export/              # （本地导出生成，已 gitignore）
```

---

## 快速开始：训练 → 导出 → 量化 → 部署

### 1）训练（N=1 与 N=5）

```bash
# 完整训练（GPU 多世界；chunk=1 输出 checkpoints/ppo_final_N1/，chunk=5 输出 checkpoints/ppo_final_N5/）
python src/train_ppo.py --steps 300000 --device cuda --name ppo_final --chunk 1 --ent-coef 0.01 --n-steps 4096
python src/train_ppo.py --steps 300000 --device cuda --name ppo_final --chunk 5 --ent-coef 0.01 --n-steps 4096
```

> 防熵坍缩提示：在 `83ms`（`0.0833s`）步长下需配合 `--n-steps 4096 --ent-coef 0.01`；导出时使用按评估成功率保存的 `best_model.zip`，不要用最终的 `ppo_model.zip`。

### 2）仿真评估（可选）

```bash
python src/eval_ppo.py --checkpoint checkpoints/ppo_final_N1/best_model.zip
python src/eval_ppo.py --checkpoint checkpoints/ppo_final_N5/best_model.zip
```

### 3）导出 ONNX + 采集校准集（同一 conda 环境内完成）

```bash
# ① 拆解 Actor 权重 → export/ppo_final_N1/、export/ppo_final_N5/
python onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N1/best_model.zip --name ppo_final_N1
python onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N5/best_model.zip --name ppo_final_N5

# ② 导出 ONNX + INT8 真实观测校准集
python onnx/export_ppo_onnx.py --actor export/ppo_final_N1/policy_actor.pt --make-calib
python onnx/export_ppo_onnx.py --actor export/ppo_final_N5/policy_actor.pt --make-calib --chunk 5
```

生成产物（`export/ppo_final_N{1,5}/`）：`actor_fp32_bs1.onnx`、`actor_fp32_bs8.onnx`（纯 `Gemm`/`Tanh`）、`calib_obs.npz`（真实观测校准集）、`policy_actor.pt`、`actor_config.yaml`。

> 注：`onnx/export_ablation.ps1` 仍保留旧 `train\` 路径未改动，请直接按上方 `python onnx/...` 命令执行。

### 4）板端构建 TensorRT 引擎（FP32 基线 + INT8 量化，务必 batch=1）

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

### 5）实车部署与测速

```bash
scp -r deploy/ppo_nav/ wheeltec@<ip>:~/wheeltec_robot/src/
# 编译并将 4 个引擎放入 models/ 后启动；完整流程见 runs/ablation_runbook_zh.md
roslaunch ppo_nav planner.launch
```

**实车 4 格消融矩阵**（详见 `runs/ablation_runbook_zh.md`）：

| 实验格 | 量化 | chunk | 引擎 | `chunk_size` | `use_prev_action` | obs 维度 |
|---|---|---|---|---|---|---|
| ① | FP32 | N=5 | `actor_fp32_n5.engine` | 5 | true | 113 |
| ② | **INT8** | **N=5（主实验）** | `actor_int8_n5.engine` | 5 | true | 113 |
| ③ | INT8 | N=1 | `actor_int8_n1.engine` | 1 | false | 105 |
| ④ | FP32 | N=1 | `actor_fp32_n1.engine` | 1 | false | 105 |

---

## 观测 / 动作 / 奖励契约（`env/wrapper.py` 中的 `IrSimEnv`）

- **观测（obs）**：
  - `lidar`：分箱取最小值再除以 `obs.lidar_range_max`（`obs.max_bins`，默认 `100` = 激光束数）
  - `goal` 极坐标：`[dist / goal_dist_norm, cos, sin]`（3 维，`goal_dist_norm` 默认 `10.0`）
  - `previous action`（`obs.include_prev_action: true`）：`N * action_dim` 维，每步 `[lin*2, (ang+1)/2]`
    - N=1：通过 `step_single` 训练，该通道始终为 0（部署时 `use_prev_action=false`）
    - N=5：chunk 训练写入上一个 chunk 的 5 步动作（部署时 `use_prev_action=true`，planner 累积 `prev_history_`）
- **动作（action）**：网络输出 μ 经 `[-1, 1]` 裁剪后映射为真实速度（差速车 `[linear_vel, angular_vel]`）。
- **奖励（reward）**（`configs/train.yaml` + `env/reward.py`）：
  - 到达：`+400.0`（`configs/train.yaml` 中的 `goal_reward`；`RewardFn` 类缺省兜底值为 `100.0`）
  - 碰撞：`-100.0`（`collision_penalty`）
  - 普通步：`lin - 0.5|ang| - 0.1 (time_penalty) - proximity_penalty + goal_shaping`（`0.5 * cos(angle_to_goal) + 1.0 * (prev_dist - dist)`）
- **终止条件**：到达 / 碰撞 / 超时（`0.0833s` 步长下 `max_steps: 1800`）。

## 更换训练/评估场景

将 `ir-sim` 场景 YAML 放入 `configs/world/`，并修改 `configs/train.yaml`：
- `src/train_ppo.py` 中当 `env.worlds` 非空时**优先使用 `env.worlds` 多世界训练，并忽略 `env.world_name`**。如需切换到单个自定义训练场景，请把 `env.worlds` 设为空列表 `[]` 并修改 `env.world_name`，或直接修改 `env.worlds` 列表。
- 修改 `env.eval_world` 指定评估场景。
- 场景要求：机器人配备 `lidar2d` 传感器，激光束数需匹配 `obs.max_bins` / `obs.lidar_len`（`100`），`range_max` 需匹配 `obs.lidar_range_max`（`7.0`），并配置 `goal`。
