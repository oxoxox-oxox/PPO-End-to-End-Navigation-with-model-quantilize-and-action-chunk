English | [中文](trt_bench_results_zh.md)

# TensorRT Three-Precision Throughput Benchmark Results

> **Provenance note**: Restored from pre-restructure benchmark records (`2026-08-23` PC & `2026-08-24` Jetson Orin Nano).
> These measurements were collected on the equivalent `1024×1024` pure-GEMM/Tanh Actor ONNX (`1.175M` MACs, N=5) using the pre-restructure `bench_trt.py` harness (`export/mlptd3_1024_N5/`). Because PPO uses the exact same `1024×1024` pure-GEMM/Tanh Actor graph (`1.175M` MACs for N=5), the TRT kernel throughput numbers directly characterize the deployed Actor network.

---

## Part 1: PC Benchmark (RTX 5070 / TRT 10.10)

> Date: 2026-08-23 ｜ PC: RTX 5070 (Blackwell `sm_120`) + Windows WDDM
> TRT `10.10.0.31` + CUDA `12.8` ｜ Calibration: 256 real obs (`IInt8EntropyCalibrator2`)
> Timing protocol: 100 warmups + 300 consecutive enqueues + `cudaDeviceSynchronize`, median over 5 runs
> Engines: `export/mlptd3_1024_N5/actor_fp32_bs{1,8,32}.onnx` and `export/mlptd3_N5/actor_fp32_bs8.onnx`
> Command (pre-restructure harness): `python eval/bench_trt.py --onnx <base> --calib <calib.npz> --batches 1,8,32`

### 1. 1024×1024 Model (Actor 1.175M MACs)

| batch | FP32 per-sample | FP16 per-sample | INT8 per-sample | INT8 vs FP32 | FP16 vs FP32 |
| --- | --- | --- | --- | --- | --- |
| 1 | 75.04 us | 71.21 us | 108.15 us | **+44.1%** (slower) | -5.1% |
| 8 | 12.88 us | 10.66 us | 9.44 us | **-26.7%** (faster) | -17.2% |
| 32 | 3.32 us | 2.16 us | 2.66 us | **-19.9%** | -34.9% |

### 2. Baseline Comparison: 400×300 Model (Actor 0.168M MACs)

| batch | FP32 | FP16 | INT8 | INT8 vs FP32 |
| --- | --- | --- | --- | --- |
| 8 | 9.49 us | 9.27 us | 9.26 us | **-2.3%** (parity) |

### 3. Output Accuracy (64 random obs, non-calibration set)

| Model | Precision pair | max-abs-diff | cos |
| --- | --- | --- | --- |
| 1024×1024 | INT8 vs FP32 | 0.000419 | 1.000000 |
| 1024×1024 | FP16 vs FP32 | 0.000418 | 1.000000 |
| 400×300 | INT8 vs FP32 | 0.000759 | 1.000000 |

### 4. PC Observations

1. **batch=1: INT8 is 44.1% slower on PC/WDDM** (launch-bound + `M=1` cannot saturate Tensor Cores).
2. **batch≥8: INT8/FP16 are 20–35% faster per sample than FP32** — quantization speedup emerges as model size grows (`0.168M` MACs at parity `-2.3%` vs `1.175M` MACs at `-26.7%` for batch=8).
3. **INT8 precision preserved**: `max-abs-diff` ~`4e-4`, `cos=1.0`.
4. Note on WDDM + GPU boost variance: single-run noise can reach ±10–35%; use multi-run medians.

---

## Part 2: Real Target Hardware (Jetson Orin Nano, 2026-08-24)

> Target: `wheeltec@<ip>` ｜ TRT `8.5.2.2` / numpy `1.23.5` ｜ Same `1024×1024` ONNX + same 256 calibration samples
> Command (pre-restructure harness): `python3 deploy/bench_trt.py`

### 1. Three Precisions × batch `{1, 8}` (`1024×1024` Model)

| batch | FP32 per-sample | FP16 per-sample | INT8 per-sample | INT8 vs FP32 | FP16 vs FP32 |
| --- | --- | --- | --- | --- | --- |
| 1 | 232.26 us | 193.76 us | 155.55 us | **-33.0%** (faster) | -16.6% |
| 8 | 38.48 us | 18.33 us | 22.42 us | **-41.7%** | **-52.4%** |

### 2. Output Accuracy (64 random obs)

| Precision pair | max-abs-diff | cos |
| --- | --- | --- |
| FP16 vs FP32 | 0.003141 | 1.000000 |
| INT8 vs FP32 | 0.068956 | 0.999954 |

(Build notes: Reshape INT64 weights cast to INT32 — harmless; FP16 build printed 4 subnormal FP16 weight warnings.)

### 3. PC (RTX 5070 / TRT 10.10 / WDDM) vs Jetson (TRT 8.5)

| Platform | bs1 INT8 vs FP32 | bs8 INT8 vs FP32 | bs8 FP16 vs FP32 | INT8 max-abs-diff |
| --- | --- | --- | --- | --- |
| PC (TRT 10.10) | **+44%** (slower) | -26.7% | -17.2% | 0.000419 |
| **Jetson (TRT 8.5)** | **-33%** (faster) | -41.7% | -52.4% | 0.068956 |

### 4. Key Takeaways

1. **"No speedup at batch=1" holds only on PC/WDDM, not on Jetson Orin Nano**: on Orin, compute dominates launch overhead even at `batch=1`, giving a real **-33.0%** latency reduction for INT8 (`155.55 us` vs `232.26 us`).
2. **batch=8**: FP16 is fastest (`-52.4%`), followed by INT8 (`-41.7%`). At **batch=1** (robot deployment mode), INT8 is fastest (`-33.0%`).
3. **Pure-GEMM large-MLP validation**: without conv/reformat casts and with `1024×1024` scale, quantization gains materialize cleanly on the embedded target.
4. **Accuracy note**: TRT 8.5 INT8 error (`max-abs 0.068956` on random obs, `cos 0.999954`) is larger than TRT 10.10 (`0.000419`); validate navigation quality directly on-robot using real-obs calibration (`build_ptq_engine.py`).
5. Single-run unlocked-clock noise exists; repeat under `nvpmodel -m 0 && jetson_clocks` in the 4-cell ablation (`runs/ablation_runbook.md`).
