[English](README.md) | 中文

# deploy/ppo_nav — PPO 实车导航部署包（ROS1 Noetic / Jetson Orin Nano / TRT 8.5）

> 本包为“INT8 量化 × Action Chunking（N=5）”的实车部署载体：PPO 策略 → ONNX → TensorRT 引擎 → 实车导航。
> 同时支持两种模型变体：**N=1**（obs 105 维，输出 `[1,2]`）与 **N=5**（obs 113 维，输出 `[1,10]`）。
> 主实验 = **N=5 + INT8**（也是 `planner.launch` 的默认配置）。完整实车测试流程见 `runs/ablation_runbook_zh.md`。

## ⚠️ 先看这里：关于 obs 构造的两条关键约定

1. **N=1 模型**通过 `IrSimEnv.step_single` 训练，训练期“上一动作”通道恒为 0 → 部署时设 `use_prev_action: false`（`chunk_size: 1`）。
2. **N=5 模型**开启 action chunking 训练，obs 中“上一动作”通道 = 上一个 chunk 执行过的 5 步动作 → 部署时设 `use_prev_action: true`（`chunk_size: 5`），由 `planner_node` 累积 `prev_history_`。

   ⚠️ `chunk_size` 与 `use_prev_action` 必须与所加载引擎严格一致（对照表见 `runs/ablation_runbook_zh.md` 第 0 节）。

## 包结构

```text
deploy/ppo_nav/
├─ package.xml / CMakeLists.txt          # catkin 包（gnu++14，TRT 8.5 + CUDA）
├─ include/ppo_nav/obs_builder.hpp       # obs 构造（N=1:105 维 / N=5:113 维，与 env/wrapper.py 对齐）
├─ include/ppo_nav/tensorrt_engine.hpp   # TRT 引擎封装
├─ src/obs_builder.cpp                   # 前方 ±90° 激光重采样归一化 + 目标极坐标 [dist/10,cos,sin]
├─ src/tensorrt_engine.cpp               # 引擎加载与同步推理（CUDA 显存缓冲）
├─ src/planner_node.cpp                  # /scan+/odom -> obs -> 推理 -> /cmd_vel_planner（chunk 开环执行）
├─ src/safety_node.cpp                   # 限幅 / 急停 / 看门狗 / 电压 / 充电标志保护 -> /cmd_vel
├─ config/params.yaml                    # 基础参数（默认对齐 N=5 主实验，launch 参数可覆盖）
├─ launch/planner.launch                 # 同时启动 planner + safety；暴露消融参数 arg
├─ launch/safety.launch                  # 单独启动 safety_node
├─ scripts/measure_node.py               # 实车测速与延迟/频率/急停统计脚本
└─ scripts/build_ptq_engine.py           # 板端构建 INT8 引擎（真实 obs 校准，导航强制 batch=1）
```

## 数据契约（严格对齐训练端）

- 输入 `obs [1,105]`（N=1）或 `[1,113]`（N=5）：
  - `[0:100]` lidar：按训练角度 `linspace(-π/2, π/2, 100)` 对前方 ±90° 最近邻重采样 100 束；`inf`/NaN/`>range_max_norm(7.0)` 截断到 `range_max_norm` 后除以 `range_max_norm`（得 `1.0`）。
  - `[100:103]` 目标极坐标：`[dist / goal_dist_norm(10), cos(a), sin(a)]`，`a = wrap(atan2(gy-y, gx-x) - yaw)`。
  - `[103:]` 上一动作通道：
    - N=1 → `[103:105]` 恒为 0（`use_prev_action: false`）
    - N=5 → `[103:113]` 上一个 chunk 的 5 步动作（`use_prev_action: true`，由 `planner_node` 累积）
- 输出 `action [1,2]`（N=1）或 `[1,10]`（N=5）：网络输出 μ（无输出 tanh）→ 节点内 `clip(μ, [-1,1])` →
  `scale_action`: `vel_min + (a+1)/2 * (vel_max-vel_min)` → `max_linear/max_angular` 限幅 → `/cmd_vel_planner`。
- 决策由**激光雷达帧驱动**（每帧新 `/scan` 触发一次决策，约 12Hz）；当 `chunk_size=5` 时单次推理生成 5 步开环动作，推理频率降为 1/5；`min_decision_period` 默认 `0.0`（匹配训练 `0.0833s` 步长）。
- `goal` 定义在 `odom` 坐标系下（相对于 `/odom` 原点）。

## 上车流程

### 1）PC 端生成导出产物（在统一 conda 环境中执行）

```bash
python onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N1/best_model.zip --name ppo_final_N1
python onnx/export_ppo_onnx.py --actor export/ppo_final_N1/policy_actor.pt --make-calib

python onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N5/best_model.zip --name ppo_final_N5
python onnx/export_ppo_onnx.py --actor export/ppo_final_N5/policy_actor.pt --make-calib --chunk 5
```

### 2）板端构建 TensorRT 引擎（务必 batch=1）

```bash
scp -r export/ppo_final_N1 export/ppo_final_N5 wheeltec@<ip>:~/ppo_deploy/
ssh wheeltec@<ip>

cd ~/ppo_deploy/ppo_final_N1
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32_n1.engine
python3 ~/wheeltec_robot/src/ppo_nav/scripts/build_ptq_engine.py \
    --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8_n1.engine --batch-size 1

cd ../ppo_final_N5
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32_n5.engine
python3 ~/wheeltec_robot/src/ppo_nav/scripts/build_ptq_engine.py \
    --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8_n5.engine --batch-size 1
```

### 3）同步代码并编译

```bash
scp -r deploy/ppo_nav/ wheeltec@<ip>:~/wheeltec_robot/src/
ssh wheeltec@<ip>
cd ~/wheeltec_robot/src/ppo_nav && find . \( -name '*.cpp' -o -name '*.hpp' \) -exec touch {} +
cd ~/wheeltec_robot
rm -rf build/ppo_nav
catkin_make -DCATKIN_WHITELIST_PACKAGES="ppo_nav" -j$(nproc)
source devel/setup.bash
catkin_make -DCATKIN_WHITELIST_PACKAGES=""

mkdir -p ~/wheeltec_robot/src/ppo_nav/models
cp ~/ppo_deploy/ppo_final_N1/actor_{fp32,int8}_n1.engine ~/wheeltec_robot/src/ppo_nav/models/
cp ~/ppo_deploy/ppo_final_N5/actor_{fp32,int8}_n5.engine ~/wheeltec_robot/src/ppo_nav/models/
```

> 编译成功判据：`catkin_make` 输出**必须出现** `Building CXX object .../planner_node.dir/...`；
> 若只看到 `Built target` + `Clock skew detected` 说明未重新编译（仍在跑旧二进制）。TRT `deprecated` 警告不影响使用。

### 4）启动（通过 `planner.launch` 参数传参；完整 4 格矩阵见 `runs/ablation_runbook_zh.md`）

> 请勿在 `roslaunch` 之前用 `rosparam set` 预设参数，因为 `planner.launch` 会重新加载 `config/params.yaml` 并按 `<arg>` 覆盖。请直接通过 `roslaunch` 参数指定各实验格配置：

```bash
# 实验格 ②（主实验）：INT8 + N=5（也是不带参数时的默认值）
roslaunch ppo_nav planner.launch \
    engine_path:=/home/wheeltec/wheeltec_robot/src/ppo_nav/models/actor_int8_n5.engine \
    chunk_size:=5 use_prev_action:=true

# 实验格 ①：FP32 + N=5
roslaunch ppo_nav planner.launch \
    engine_path:=/home/wheeltec/wheeltec_robot/src/ppo_nav/models/actor_fp32_n5.engine \
    chunk_size:=5 use_prev_action:=true

# 实验格 ③：INT8 + N=1
roslaunch ppo_nav planner.launch \
    engine_path:=/home/wheeltec/wheeltec_robot/src/ppo_nav/models/actor_int8_n1.engine \
    chunk_size:=1 use_prev_action:=false

# 实验格 ④：FP32 + N=1
roslaunch ppo_nav planner.launch \
    engine_path:=/home/wheeltec/wheeltec_robot/src/ppo_nav/models/actor_fp32_n1.engine \
    chunk_size:=1 use_prev_action:=false

# planner.launch 已包含 safety_node；如需单独启动安全节点：
roslaunch ppo_nav safety.launch
```

### 5）实车验证顺序

1. 抬起车轮或保持驻车状态确认启动日志：
   - N=5：`[TRT] engine loaded: input 113 elements, output 10 elements` 且 `use_prev_action=1`
   - N=1：`[TRT] engine loaded: input 105 elements, output 2 elements` 且 `use_prev_action=0`
   - 确认无 `engine input=... (expect ...)` 维度不匹配警告，也无 `engine forward failed`。
2. 落地前通过 RViz 或原始 `/scan` 确认激光方向（`reverse_scan`）及 `laser -> base_link` TF 朝向是否正确。
3. 使用 `max_linear:=0.3` 低速试跑（可选加 `debug_log:=true` 观察 `[planner] infer done. obs=... act=...` 与输出速度 `v=(lin, ang) chunk_remain=...`），确认障碍物进入 `stop_dist` 时 `safety_node` 能可靠触发急停，再逐步增加障碍物。

## 日志判读

**正常**（N=1）：
```text
[TRT] engine loaded: input 105 elements, output 2 elements
[planner] ready. goal=(5.00, 5.00), scan-driven, min_period=0.00s, use_prev_action=0
```
**正常**（N=5）：
```text
[TRT] engine loaded: input 113 elements, output 10 elements
[planner] ready. goal=(5.00, 5.00), scan-driven, min_period=0.00s, use_prev_action=1
```
**异常维度不匹配（示例）**：
- `[planner] engine input=840 (expect 105 for chunk=1)`（或 `904 (expect 113 for chunk=5)`）：误用了 `bs8` 引擎 → 返回第 2 步使用 `--batch-size 1` 重建。
- `[planner] engine input=113 (expect 105 for chunk=1)`：`chunk_size` 与所加载引擎不匹配 → 检查并传入对应的 `chunk_size:=` 和 `use_prev_action:=`。

## 参数说明（`config/params.yaml` + `launch/planner.launch`）

- **可通过 `planner.launch` 参数直接覆盖**：
  `engine_path`、`chunk_size`（N=5 填 `5` / N=1 填 `1`）、`use_prev_action`（N=5 填 `true` / N=1 填 `false`）、`max_linear`（默认 `0.8`，初测建议 `0.3`）、`max_angular`（`1.0`）、`goal_x`（`5.0`，`odom` 系）、`goal_y`（`5.0`，`odom` 系）、`min_decision_period`（`0.0`，激光帧驱动约 12Hz，对齐训练 `step_time=0.0833s`）、`debug_log`（`false`）。
- **在 `config/params.yaml` 中配置**：
  `scan_timeout`（`0.5`）、`num_beams`（`100`）、`range_max_norm`（`7.0`）、`goal_dist_norm`（`10.0`）、`half_fov_deg`（`90.0`）、`reverse_scan`（`false`）、`laser_yaw_fallback_deg`（`180.0`）、`vel_min`/`vel_max`（`[-1.0, -1.0]` / `[1.0, 1.0]`），以及 `safety` 段（`stop_dist`、`slow_dist`、`slow_factor`、`min_voltage`、`watchdog_timeout`、`max_linear`、`max_angular`、`front_only`）。

## 踩坑速查

- **引擎 batch**：实车导航必须用 batch=1 引擎（`--batch-size 1`）；误用 `bs8` 引擎会导致维度不匹配（`840` 或 `904`）而停车。
- **launch 参数覆盖**：请通过 `roslaunch` 传 `chunk_size:=` 和 `use_prev_action:=`，不要在 `roslaunch` 前用 `rosparam set`。
- **时钟偏移跳过编译**：`scp` 可能导致文件修改时间超前 → 编译前务必先 `touch` 源码（见第 3 步）。
- **CATKIN_WHITELIST_PACKAGES**：单包编译完记得用 `""` 清空，恢复全工作空间编译。
- **gnu++14**：使用 `-std=c++14` 会隐藏 `M_PI` 导致 `planner_node.cpp` / `safety_node.cpp` 编译失败，需保留 `CMakeLists.txt` 中的 `gnu++14`。TRT deprecated 警告可忽略。
- **命令行 `$(find)`**：bash 会先展开 `$()` → 传 `engine_path:=` 时使用绝对路径或单引号。
- **TF32**：构建 FP32 基线引擎时 `trtexec` 必须加 `--noTF32` 才是纯 FP32。
