English | [中文](README_zh.md)

# deploy/ppo_nav — PPO Real-Robot Navigation Deployment Package (ROS1 Noetic / Jetson Orin Nano / TRT 8.5)

> This package is the real-robot carrier for **INT8 quantization × action chunking (N=5)**: PPO policy → ONNX → TensorRT engine → real-robot navigation.
> It supports two model variants: **N=1** (obs 105 dims, output `[1,2]`) and **N=5** (obs 113 dims, output `[1,10]`).
> Main experiment = **N=5 + INT8** (default of `planner.launch`). Full on-robot benchmarking protocol is in `runs/ablation_runbook.md`.

## ⚠️ Read first: two key facts about obs construction

1. **N=1 models** are trained through `IrSimEnv.step_single`, so the "previous action" channel is always 0 → deployment uses `use_prev_action: false` (`chunk_size: 1`).
2. **N=5 models** are trained with action chunking; the obs "previous action" channel = the previous chunk's 5 actions → deployment uses `use_prev_action: true` (`chunk_size: 5`), with `planner_node` accumulating `prev_history_`.

   ⚠️ Both `chunk_size` and `use_prev_action` must strictly match the engine loaded (see the matrix in `runs/ablation_runbook.md` section 0).

## Package Structure

```text
deploy/ppo_nav/
├─ package.xml / CMakeLists.txt          # catkin package (gnu++14, TRT 8.5 + CUDA)
├─ include/ppo_nav/obs_builder.hpp       # obs construction (N=1:105 dims / N=5:113 dims, matches env/wrapper.py)
├─ include/ppo_nav/tensorrt_engine.hpp   # TRT engine wrapper
├─ src/obs_builder.cpp                   # lidar forward resampling/normalization + goal [dist/10,cos,sin]
├─ src/tensorrt_engine.cpp               # engine load/forward (CUDA buffers, synchronous)
├─ src/planner_node.cpp                  # /scan+/odom -> obs -> inference -> /cmd_vel_planner (chunk open-loop execution)
├─ src/safety_node.cpp                   # clipping/emergency-stop/watchdog/battery/charging -> /cmd_vel
├─ config/params.yaml                    # base parameters (defaults aligned with N=5 main experiment; launch args override)
├─ launch/planner.launch                 # starts planner + safety together; exposes ablation args
├─ launch/safety.launch                  # starts safety_node alone
├─ scripts/measure_node.py               # real-robot benchmarking measurement (latency/frequency/emergency-stop)
└─ scripts/build_ptq_engine.py           # build INT8 engine on board (real-obs calibration, batch=1 for navigation)
```

## Data Contract (exact obs / action format)

- Input `obs [1,105]` (N=1) or `[1,113]` (N=5):
  - `[0:100]` lidar: resampled to 100 beams across forward ±90° at the training angles (`linspace(-π/2, π/2, 100)`), inf/NaN/`>range_max_norm(7.0)` clipped to `range_max_norm` and divided by `range_max_norm` (`1.0`).
  - `[100:103]` goal polar coordinates: `[dist / goal_dist_norm(10), cos(a), sin(a)]`, `a = wrap(atan2(gy-y, gx-x) - yaw)`.
  - `[103:]` previous action:
    - N=1 → `[103:105]` always 0 (`use_prev_action: false`)
    - N=5 → `[103:113]` previous chunk's 5 actions (`use_prev_action: true`, accumulated by `planner_node`)
- Output `action [1,2]` (N=1) or `[1,10]` (N=5): μ (PPO has no output tanh) → in-node `clip(μ, [-1,1])` →
  `scale_action`: `vel_min + (a+1)/2 * (vel_max-vel_min)` → `max_linear/max_angular` clamps → `/cmd_vel_planner`.
- Decisions are **driven by the laser frames** (each new `/scan` triggers one decision tick, ~12Hz); with `chunk_size=5`, a single inference produces 5 open-loop steps, cutting inference frequency to 1/5; optional `min_decision_period` throttling (`0.0` by default to match `0.0833s` training).
- `goal` is defined in the `odom` frame (relative to `/odom` origin).

## On-Robot Procedure

### 1) PC-side artifacts (with the unified conda env activated)

```bash
python onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N1/best_model.zip --name ppo_final_N1
python onnx/export_ppo_onnx.py --actor export/ppo_final_N1/policy_actor.pt --make-calib

python onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N5/best_model.zip --name ppo_final_N5
python onnx/export_ppo_onnx.py --actor export/ppo_final_N5/policy_actor.pt --make-calib --chunk 5
```

### 2) Build engines on board (batch=1!)

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

### 3) Sync package and build

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

> Build success criterion: the `catkin_make` output **must contain** `Building CXX object .../planner_node.dir/...`;
> if you only see `Built target` + `Clock skew detected` = not recompiled (old binary). TRT `deprecated` warnings are normal.

### 4) Launch (pass parameters via `planner.launch` args; full 4-cell matrix in `runs/ablation_runbook.md`)

> Do NOT use `rosparam set` before `roslaunch`, because `planner.launch` loads `config/params.yaml` first and then applies its `<arg>` overrides. Pass cell parameters directly as `roslaunch` args:

```bash
# Cell ② (main experiment): INT8 + N=5 (also the default when launched with no args)
roslaunch ppo_nav planner.launch \
    engine_path:=/home/wheeltec/wheeltec_robot/src/ppo_nav/models/actor_int8_n5.engine \
    chunk_size:=5 use_prev_action:=true

# Cell ①: FP32 + N=5
roslaunch ppo_nav planner.launch \
    engine_path:=/home/wheeltec/wheeltec_robot/src/ppo_nav/models/actor_fp32_n5.engine \
    chunk_size:=5 use_prev_action:=true

# Cell ③: INT8 + N=1
roslaunch ppo_nav planner.launch \
    engine_path:=/home/wheeltec/wheeltec_robot/src/ppo_nav/models/actor_int8_n1.engine \
    chunk_size:=1 use_prev_action:=false

# Cell ④: FP32 + N=1
roslaunch ppo_nav planner.launch \
    engine_path:=/home/wheeltec/wheeltec_robot/src/ppo_nav/models/actor_fp32_n1.engine \
    chunk_size:=1 use_prev_action:=false

# safety_node is already included in planner.launch; or launch alone:
roslaunch ppo_nav safety.launch
```

### 5) On-vehicle verification order

1. Confirm startup logs with handbrake released/wheels off ground:
   - N=5: `[TRT] engine loaded: input 113 elements, output 10 elements` and `use_prev_action=1`
   - N=1: `[TRT] engine loaded: input 105 elements, output 2 elements` and `use_prev_action=0`
   - Confirm no `engine input=... (expect ...)` mismatch warning and no `engine forward failed`.
2. Check `/scan` direction (`reverse_scan`) and `laser -> base_link` TF transform in RViz or raw `/scan` messages before moving.
3. Run a low-speed trial with `max_linear:=0.3` (optionally `debug_log:=true` to inspect `[planner] infer done. obs=... act=...` and published `v=(lin, ang) chunk_remain=...`), verify `safety_node` emergency-stop triggers inside `stop_dist`, then add obstacles gradually.

## Log Interpretation

**Normal** (N=1):
```text
[TRT] engine loaded: input 105 elements, output 2 elements
[planner] ready. goal=(5.00, 5.00), scan-driven, min_period=0.00s, use_prev_action=0
```
**Normal** (N=5):
```text
[TRT] engine loaded: input 113 elements, output 10 elements
[planner] ready. goal=(5.00, 5.00), scan-driven, min_period=0.00s, use_prev_action=1
```
**Abnormal dimension mismatch** (examples):
- `[planner] engine input=840 (expect 105 for chunk=1)` (or `904 (expect 113 for chunk=5)`): a `bs8` engine was loaded instead of `bs1` → rebuild with `--batch-size 1`.
- `[planner] engine input=113 (expect 105 for chunk=1)`: `chunk_size` does not match the loaded engine → pass matching `chunk_size:=` and `use_prev_action:=` to `planner.launch`.

## Parameters (`config/params.yaml` + `launch/planner.launch`)

- **Overridable via `planner.launch` args**:
  `engine_path`, `chunk_size` (`5` for N=5 / `1` for N=1), `use_prev_action` (`true` for N=5 / `false` for N=1), `max_linear` (`0.8`, use `0.3` for initial tests), `max_angular` (`1.0`), `goal_x` (`5.0`, `odom` frame), `goal_y` (`5.0`, `odom` frame), `min_decision_period` (`0.0`, scan-frame-driven ~12Hz, matches training `step_time=0.0833s`), `debug_log` (`false`).
- **Configured in `config/params.yaml`**:
  `scan_timeout` (`0.5`), `num_beams` (`100`), `range_max_norm` (`7.0`), `goal_dist_norm` (`10.0`), `half_fov_deg` (`90.0`), `reverse_scan` (`false`), `laser_yaw_fallback_deg` (`180.0`), `vel_min`/`vel_max` (`[-1.0, -1.0]` / `[1.0, 1.0]`), and `safety` section (`stop_dist`, `slow_dist`, `slow_factor`, `min_voltage`, `watchdog_timeout`, `max_linear`, `max_angular`, `front_only`).

## Pitfall Quick Reference

- **Engine batch**: navigation always uses batch=1 engines (`--batch-size 1` is mandatory); a `bs8` engine causes input size mismatch (`840` or `904`) and stops the robot.
- **Launch overrides**: always pass `chunk_size:=` and `use_prev_action:=` via `roslaunch` args rather than `rosparam set` before `roslaunch`.
- **Clock skew skips compilation**: `scp` can carry future mtimes → `touch` the sources before building (see step 3).
- **CATKIN_WHITELIST_PACKAGES**: after building a single package, reset with `""` to restore the whole workspace.
- **gnu++14**: `-std=c++14` hides `M_PI` and causes compilation failure in `planner_node.cpp` / `safety_node.cpp`; keep `gnu++14` in `CMakeLists.txt`. TRT deprecated warnings are harmless.
- **Command-line `$(find)`**: bash expands `$()` first → use an absolute path or single quotes for `engine_path:=` at launch.
- **TF32**: FP32 baseline engines must pass `--noTF32` to `trtexec` to be true FP32.
