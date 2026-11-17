[English](ablation_runbook.md) | 中文

# 实车 4 格消融实验 · 上车操作手册（量化 × Action Chunk）

> 目标：在 Jetson Orin Nano 上完成 4 格对照实验（量化开/关 × chunk 开/关），
> 测量 scan→决策延时、单次推理耗时、推理频率、动作频率、安全急停次数与 GPU 占用。
> **主实验 = 实验格 ②（INT8 × N=5）**；对应模型为 `ppo_final_N1`（单步）与 `ppo_final_N5`（chunk=5）。

---

## 0. 四格实验矩阵与参数对照

| 实验格 | 量化 | chunk | 引擎 | `chunk_size` | `use_prev_action` | obs 维度 |
| --- | --- | --- | --- | --- | --- | --- |
| ① | OFF FP32 | N=5 | `actor_fp32_n5.engine` | 5 | true | 113 |
| ② | ON INT8 | N=5 | `actor_int8_n5.engine` | 5 | true | 113 |
| ③ | ON INT8 | N=1 | `actor_int8_n1.engine` | 1 | false | 105 |
| ④ | OFF FP32 | N=1 | `actor_fp32_n1.engine` | 1 | false | 105 |

---

## 1. PC 端生成导出产物（训练后在本地生成，仓库未内置）

在统一 PC conda 环境中执行：

```bash
# N=1（obs 105，act 2）
python onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N1/best_model.zip --name ppo_final_N1
python onnx/export_ppo_onnx.py --actor export/ppo_final_N1/policy_actor.pt --make-calib

# N=5（obs 113，act 10）
python onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N5/best_model.zip --name ppo_final_N5
python onnx/export_ppo_onnx.py --actor export/ppo_final_N5/policy_actor.pt --make-calib --chunk 5
```

每个 `export/ppo_final_N{1,5}/` 目录包含：`actor_fp32_bs1.onnx`、`actor_fp32_bs8.onnx`、`calib_obs.npz`、`policy_actor.pt`、`actor_config.yaml`。
板端 INT8 引擎构建使用仓库内的 `deploy/ppo_nav/scripts/build_ptq_engine.py`。

---

## 2. 板端构建引擎（先同步，后构建）

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

> 导航全程使用 `batch=1`；FP32 基线必须加 `--noTF32`。完成后将 4 个 `.engine` 拷入 `~/wheeltec_robot/src/ppo_nav/models/`。

---

## 3. 编译部署包

```bash
scp -r deploy/ppo_nav/ wheeltec@<ip>:~/wheeltec_robot/src/
ssh wheeltec@<ip>
cd ~/wheeltec_robot/src/ppo_nav && find . \( -name '*.cpp' -o -name '*.hpp' \) -exec touch {} +
cd ~/wheeltec_robot && rm -rf build/ppo_nav
catkin_make -DCATKIN_WHITELIST_PACKAGES="ppo_nav" -j$(nproc)
source devel/setup.bash
catkin_make -DCATKIN_WHITELIST_PACKAGES=""
```

> 判据：输出中必须看到 `Building CXX object ... planner_node`。TRT deprecated 警告可忽略。

---

## 4. 基准测量流程（每个实验格重复执行）

1. **锁频**：`sudo nvpmodel -m 0 && sudo jetson_clocks`
2. **启动测量节点**：

   ```bash
   # 终端 A：GPU/算力占用（全程运行，结束后 Ctrl+C 保存）
   tegrastats --interval 500 --logfile /home/wheeltec/gpu_<cell>.log
   # 终端 B：延迟 / 频率 / 急停统计节点
   python3 ~/wheeltec_robot/src/ppo_nav/scripts/measure_node.py _out:=/home/wheeltec/measure_<cell>.csv
   ```

3. **启动导航**（通过 `planner.launch` 参数传 `engine_path`、`chunk_size`、`use_prev_action` 与 `max_linear`；不要在 `roslaunch` 前用 `rosparam set`，否则会被 `planner.launch` 加载 `config/params.yaml` 覆盖）：

   ```bash
   # 实验格 ②（主实验）：INT8 + N=5
   roslaunch ppo_nav planner.launch \
       engine_path:=/home/wheeltec/wheeltec_robot/src/ppo_nav/models/actor_int8_n5.engine \
       chunk_size:=5 use_prev_action:=true max_linear:=0.3

   # 实验格 ①：FP32 + N=5
   # engine_path:=.../actor_fp32_n5.engine chunk_size:=5 use_prev_action:=true max_linear:=0.3

   # 实验格 ③：INT8 + N=1
   # engine_path:=.../actor_int8_n1.engine chunk_size:=1 use_prev_action:=false max_linear:=0.3

   # 实验格 ④：FP32 + N=1
   # engine_path:=.../actor_fp32_n1.engine chunk_size:=1 use_prev_action:=false max_linear:=0.3
   ```

4. **固定测试跑道**：终点设在 `(5, 5)`（`odom` 坐标系），在车前 1–2m 处放置障碍物，低速（`max_linear:=0.3`）运行，每格重复 **K 次**（建议 K ≥ 10），人工记录每次到达 / 碰撞 / 安全急停情况。
5. **结束保存**：Ctrl+C 停止导航和测量脚本，保留 `measure_<cell>.csv` 与 `gpu_<cell>.log`。

---

## 5. 每个实验格需保留的数据

> 说明：`measure_node.py` 导出的 `measure_*.csv` 为三列长表 `t, kind, value_ms`（其中 `kind` 取值为 `"scan_age_ms"` 或 `"fwd_ms"`），退出时终端打印 `n / mean / median / max`。

| 数据项 | 来源 |
| --- | --- |
| scan→决策延时 | `measure_*.csv` 中 `kind == "scan_age_ms"` 的行（汇总输出 `mean / median / max`） |
| 推理耗时（量化直接作用项） | `measure_*.csv` 中 `kind == "fwd_ms"` 的行（汇总输出 `mean / median / max`） |
| 推理频率 | `measure_node.py` 退出汇总中的 `inference Hz` |
| 动作发布频率 | `measure_node.py` 退出汇总中的 `action Hz` |
| 安全急停次数 | `measure_node.py` 退出汇总中的 `safety stops` |
| GPU 占用 | `gpu_*.log`（`tegrastats` 中的 `GR3D` 占用） |
| 碰撞 / 到达结果 | 第 4.4 步人工记录 |

---

## 6. 实车验证顺序

1. 确认启动日志：
   - N=5：`[TRT] engine loaded: input 113 elements, output 10 elements` 且 `use_prev_action=1`
   - N=1：`[TRT] engine loaded: input 105 elements, output 2 elements` 且 `use_prev_action=0`
   - 确认无 `engine input=... (expect ...)` 维度不匹配警告，且无 `engine forward failed`。
2. 起步前通过 RViz 或原始 `/scan` 确认激光方向（`reverse_scan`）与 `laser -> base_link` TF 朝向。
3. 使用 `max_linear:=0.3` 低速试跑（可选开启 `debug_log:=true` 观察 `[planner] infer done. obs=... act=...` 与 `v=(lin, ang) chunk_remain=...`），确认障碍进入 `stop_dist` 时 `safety_node` 触发急停，随后逐步增加障碍物。
