English | [中文](ablation_runbook_zh.md)

# Real-Robot 4-Cell Ablation · On-Robot Runbook (Quantization × Action Chunk)

> Goal: on a Jetson Orin Nano, run 4 cells (quantization ON/OFF × chunk ON/OFF),
> measuring scan→decision latency, inference time, inference frequency, action frequency, safety emergency-stops, and GPU usage.
> **Main experiment = cell ② (INT8 × N=5)**; corresponding models: `ppo_final_N1` (single-step) and `ppo_final_N5` (chunk=5).

---

## 0. Four-Cell Matrix and Parameters

| Cell | Quantization | chunk | Engine | `chunk_size` | `use_prev_action` | obs dim |
| --- | --- | --- | --- | --- | --- | --- |
| ① | OFF FP32 | N=5 | `actor_fp32_n5.engine` | 5 | true | 113 |
| ② | ON INT8 | N=5 | `actor_int8_n5.engine` | 5 | true | 113 |
| ③ | ON INT8 | N=1 | `actor_int8_n1.engine` | 1 | false | 105 |
| ④ | OFF FP32 | N=1 | `actor_fp32_n1.engine` | 1 | false | 105 |

---

## 1. PC-Side Artifacts (generated locally after training; not bundled in repo)

Run inside the unified PC conda environment:

```bash
# N=1 (obs 105, act 2)
python onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N1/best_model.zip --name ppo_final_N1
python onnx/export_ppo_onnx.py --actor export/ppo_final_N1/policy_actor.pt --make-calib

# N=5 (obs 113, act 10)
python onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N5/best_model.zip --name ppo_final_N5
python onnx/export_ppo_onnx.py --actor export/ppo_final_N5/policy_actor.pt --make-calib --chunk 5
```

Each `export/ppo_final_N{1,5}/` contains: `actor_fp32_bs1.onnx`, `actor_fp32_bs8.onnx`, `calib_obs.npz`, `policy_actor.pt`, `actor_config.yaml`.
The on-board INT8 engine build script uses the repo-tracked `deploy/ppo_nav/scripts/build_ptq_engine.py`.

---

## 2. Build Engines On Board (sync first, then build)

```bash
scp -r export/ppo_final_N1 export/ppo_final_N5 wheeltec@<ip>:~/ppo_deploy/
ssh wheeltec@<ip>

cd ~/ppo_deploy/ppo_final_N1
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32_n1.engine
python3 ~/wheeltec_robot/src/ppo_nav/scripts/build_ptq_engine.py --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8_n1.engine --batch-size 1

cd ../ppo_final_N5
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32_n5.engine
python3 ~/wheeltec_robot/src/ppo_nav/scripts/build_ptq_engine.py --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8_n5.engine --batch-size 1
```

> Navigation always uses `batch=1`; `--noTF32` is required for a true FP32 baseline. Copy the 4 `.engine` files into
> `~/wheeltec_robot/src/ppo_nav/models/`.

---

## 3. Build the Deployment Package

```bash
scp -r deploy/ppo_nav/ wheeltec@<ip>:~/wheeltec_robot/src/
ssh wheeltec@<ip>
cd ~/wheeltec_robot/src/ppo_nav && find . \( -name '*.cpp' -o -name '*.hpp' \) -exec touch {} +
cd ~/wheeltec_robot && rm -rf build/ppo_nav
catkin_make -DCATKIN_WHITELIST_PACKAGES="ppo_nav" -j$(nproc)
source devel/setup.bash
catkin_make -DCATKIN_WHITELIST_PACKAGES=""
```

> Criterion: the output must contain `Building CXX object ... planner_node`. TRT deprecated warnings are normal.

---

## 4. Benchmarking Protocol (performed for every cell)

1. **Lock frequency**: `sudo nvpmodel -m 0 && sudo jetson_clocks`
2. **Start measurement**:

   ```bash
   # Terminal A: GPU/compute (keep running throughout, Ctrl+C to save after)
   tegrastats --interval 500 --logfile /home/wheeltec/gpu_<cell>.log
   # Terminal B: latency/frequency/emergency-stop measurement node
   python3 ~/wheeltec_robot/src/ppo_nav/scripts/measure_node.py _out:=/home/wheeltec/measure_<cell>.csv
   ```

3. **Launch navigation** (pass `engine_path`, `chunk_size`, `use_prev_action`, and `max_linear` via `planner.launch` args; do **not** rely on `rosparam set` before `roslaunch`, because `planner.launch` loads `config/params.yaml` first):

   ```bash
   # Cell ② (main experiment): INT8 + N=5
   roslaunch ppo_nav planner.launch \
       engine_path:=/home/wheeltec/wheeltec_robot/src/ppo_nav/models/actor_int8_n5.engine \
       chunk_size:=5 use_prev_action:=true max_linear:=0.3

   # Cell ①: FP32 + N=5
   # engine_path:=.../actor_fp32_n5.engine chunk_size:=5 use_prev_action:=true max_linear:=0.3

   # Cell ③: INT8 + N=1
   # engine_path:=.../actor_int8_n1.engine chunk_size:=1 use_prev_action:=false max_linear:=0.3

   # Cell ④: FP32 + N=1
   # engine_path:=.../actor_fp32_n1.engine chunk_size:=1 use_prev_action:=false max_linear:=0.3
   ```

4. **Fixed test lane**: goal at `(5, 5)` (`odom` frame), place an obstacle 1–2m ahead of the robot, run at low speed (`max_linear:=0.3`), repeat **K times** per cell (recommended K ≥ 10), recording each trial's outcome (arrival / collision / safety stop).
5. **Finish**: Ctrl+C to stop navigation and measurement, save `measure_<cell>.csv` and `gpu_<cell>.log`.

---

## 5. Data to Keep Per Cell

> `measure_node.py` writes `measure_*.csv` in long format with columns `t, kind, value_ms` (where `kind` is `"scan_age_ms"` or `"fwd_ms"`), and prints `n / mean / median / max` on exit.

| Data | Source |
| --- | --- |
| scan→decision latency | `measure_*.csv` rows where `kind == "scan_age_ms"` (summary prints `mean / median / max`) |
| Inference time (quantization item) | `measure_*.csv` rows where `kind == "fwd_ms"` (summary prints `mean / median / max`) |
| Inference frequency | `inference Hz` from `measure_node.py` exit summary |
| Action frequency | `action Hz` from `measure_node.py` exit summary |
| Safety emergency-stops | `safety stops` from `measure_node.py` exit summary |
| GPU usage | `gpu_*.log` (`GR3D` usage from `tegrastats`) |
| Collisions / arrivals | manually recorded in step 4.4 |

---

## 6. On-Vehicle Verification Order

1. Confirm startup log:
   - N=5: `[TRT] engine loaded: input 113 elements, output 10 elements` and `use_prev_action=1`
   - N=1: `[TRT] engine loaded: input 105 elements, output 2 elements` and `use_prev_action=0`
   - Confirm no `engine input=... (expect ...)` mismatch warning and no `engine forward failed`.
2. Verify `/scan` direction (`reverse_scan`) and `laser -> base_link` TF orientation in RViz or raw `/scan` before driving.
3. Run low-speed trials with `max_linear:=0.3` (optionally `debug_log:=true` to inspect `[planner] infer done. obs=... act=...` and `v=(lin, ang) chunk_remain=...`), verify `safety_node` emergency-stop triggers inside `stop_dist`, and add obstacles gradually.
