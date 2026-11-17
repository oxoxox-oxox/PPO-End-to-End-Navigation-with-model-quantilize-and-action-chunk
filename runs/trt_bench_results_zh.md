[English](trt_bench_results.md) | 中文

# TRT 三精度吞吐基准结果

> **来源说明**：恢复自仓库重构前的历史实测记录（`2026-08-23` PC 与 `2026-08-24` Jetson Orin Nano）。
> 该组数据使用重构前的 `bench_trt.py` 测速脚本对同结构 `1024×1024` 纯 GEMM/Tanh Actor ONNX（`1.175M` MACs，N=5，`export/mlptd3_1024_N5/`）实测得到。由于当前 PPO 策略采用完全相同的 `1024×1024` 纯 GEMM/Tanh 网络拓扑（N=5 时同样约 `1.175M` MACs），其 TRT 算子与端到端核函数耗时可直接反映部署模型的推理性能。

---

## 一、PC 端基准（RTX 5070 / TRT 10.10）

> 实测日期：2026-08-23 ｜ PC：RTX 5070（Blackwell `sm_120`）+ Windows WDDM
> TRT `10.10.0.31` + CUDA `12.8` ｜ 校准：真实 obs 256 条（`IInt8EntropyCalibrator2`）
> 测速口径：warmup 100 次 + 连续 enqueue 300 次 + `cudaDeviceSynchronize`，5 轮取中位数
> 引擎：`export/mlptd3_1024_N5/actor_fp32_bs{1,8,32}.onnx` 与 `export/mlptd3_N5/actor_fp32_bs8.onnx`
> 复现命令（重构前测速脚本）：`python eval/bench_trt.py --onnx <基底> --calib <calib.npz> --batches 1,8,32`

### 1. 1024×1024 模型（Actor 1.175M MACs）

| batch | FP32 per-sample | FP16 per-sample | INT8 per-sample | INT8 vs FP32 | FP16 vs FP32 |
| --- | --- | --- | --- | --- | --- |
| 1 | 75.04 us | 71.21 us | 108.15 us | **+44.1%**（更慢） | -5.1% |
| 8 | 12.88 us | 10.66 us | 9.44 us | **-26.7%**（更快） | -17.2% |
| 32 | 3.32 us | 2.16 us | 2.66 us | **-19.9%** | -34.9% |

### 2. 对照：400×300 模型（Actor 0.168M MACs）

| batch | FP32 | FP16 | INT8 | INT8 vs FP32 |
| --- | --- | --- | --- | --- |
| 8 | 9.49 us | 9.27 us | 9.26 us | **-2.3%**（打平） |

### 3. 输出精度对比（64 条随机 obs，非校准数据）

| 模型 | 精度对 | max-abs-diff | cos |
| --- | --- | --- | --- |
| 1024×1024 | INT8 vs FP32 | 0.000419 | 1.000000 |
| 1024×1024 | FP16 vs FP32 | 0.000418 | 1.000000 |
| 400×300 | INT8 vs FP32 | 0.000759 | 1.000000 |

### 4. PC 端结论

1. **batch=1：INT8 反而慢 44.1%**（launch-bound + `M=1` 无法利用 Tensor Core）。
2. **batch≥8：INT8/FP16 相对 FP32 每样本快 20–35%**——量化优势随模型变大而显现：`0.168M` MACs 时三档打平（`-2.3%`），`1.175M` MACs 时 INT8 快 `26.7%`（batch=8）。
3. **INT8 精度无损**：`max-abs-diff` ~`4e-4`，`cos=1.0`。
4. 测量噪声提示（WDDM + GPU boost）：单次跑差异可达 ±10–35%，定性结论稳定，建议多轮取中位数。

---

## 二、真机实测（Jetson Orin Nano，2026-08-24）

> 实测：`wheeltec@<ip>` ｜ TRT `8.5.2.2` / numpy `1.23.5` ｜ 同 `1024×1024` 模型 ONNX + 同 256 条校准
> 复现命令（重构前测速脚本）：`python3 deploy/bench_trt.py`

### 1. 三精度 × batch `{1, 8}`（`1024×1024` 模型）

| batch | FP32 per-sample | FP16 per-sample | INT8 per-sample | INT8 vs FP32 | FP16 vs FP32 |
| --- | --- | --- | --- | --- | --- |
| 1 | 232.26 us | 193.76 us | 155.55 us | **-33.0%**（更快） | -16.6% |
| 8 | 38.48 us | 18.33 us | 22.42 us | **-41.7%** | **-52.4%** |

### 2. 输出精度对比（64 条随机 obs）

| 精度对 | max-abs-diff | cos |
| --- | --- | --- |
| FP16 vs FP32 | 0.003141 | 1.000000 |
| INT8 vs FP32 | 0.068956 | 0.999954 |

（构建警告：Reshape 的 INT64 权重 cast 到 INT32——无害；FP16 构建有 4 个 subnormal FP16 权重警告。）

### 3. 与 PC（RTX 5070 / TRT 10.10 / WDDM）对比

| 平台 | bs1 INT8 vs FP32 | bs8 INT8 vs FP32 | bs8 FP16 vs FP32 | INT8 max-abs-diff |
| --- | --- | --- | --- | --- |
| PC（TRT 10.10） | **+44%**（更慢） | -26.7% | -17.2% | 0.000419 |
| **Jetson（TRT 8.5）** | **-33%**（更快） | -41.7% | -52.4% | 0.068956 |

### 4. 关键结论

1. **“batch=1 量化无收益”仅在 PC/WDDM 成立；在目标机 Jetson Orin 上不成立**：Orin 算力相对 launch 开销占比更高，INT8 在 `batch=1` 下即可带来 **-33.0%** 延迟降低（`155.55 us` vs `232.26 us`）。
2. **batch=8**：FP16 最优（`-52.4%`），INT8 次之（`-41.7%`）；**batch=1**（实车导航模式）下 INT8 最优（`-33.0%`）。
3. **纯 GEMM 大 MLP 设计得到验证**：去掉卷积与 Reformat cast 并放大至 `1024×1024` 后，量化收益在真机端稳定显现。
4. **精度提示**：TRT 8.5 的 INT8 量化误差（随机 obs 上 `max-abs 0.068956`，`cos 0.999954`）高于 TRT 10.10（`0.000419`），实车应配合真实 obs 校准集构建引擎（`build_ptq_engine.py`）并结合实车导航验证。
5. 未锁频单次测速存在波动；实车 4 格消融实验请在 `nvpmodel -m 0 && jetson_clocks` 下复测（见 `runs/ablation_runbook_zh.md`）。
