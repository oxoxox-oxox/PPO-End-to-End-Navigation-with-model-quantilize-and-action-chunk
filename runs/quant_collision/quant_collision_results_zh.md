[English](quant_collision_results.md) | 中文

# INT8 量化 vs FP32 碰撞率对照实验（ir-sim 仿真）

> **适用范围说明**：本结果由 `src/eval_quant_collision.py` 生成，采用 PyTorch 对称 per-tensor fake-quant 模拟（**非**车端 TensorRT 引擎），测试对象为早期 N=1 模型 `checkpoints/ppo_mw_N1/best_model.zip`（仓库未内置），运行于 `eval_world_noiseless.yaml`（`step_time: 0.3s`、`max_steps: 500`）。`src/eval_quant_collision.py` 仅支持 N=1 单步模型，仓库中无单独的 N=5 量化碰撞对比脚本。

- checkpoint：`checkpoints/ppo_mw_N1/best_model.zip`
- 校准集：`export/ppo_mw/calib_obs.npz`
- 回合数：300（配对对比，相同起点 / 无激光噪声 / 静态障碍物 `eval_world_noiseless.yaml`）

## 输出一致性（256 条校准集）

| max-abs-diff | mean-abs-diff | cos |
|---|---|---|
| 0.00875 | 0.00185 | 0.999879693 |

## 碰撞率 / 成功率

| 条件 | success | collision | timeout |
|---|---|---|---|
| fp32 | 41.67% | 17.00% | 41.33% |
| int8 | 41.33% | 18.00% | 40.67% |

净效应（量化后 碰撞→成功 减去 成功→碰撞）：-1 回合 / 300 回合
平均首步动作 max-abs 扰动：0.00327
