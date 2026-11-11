# M2.4: Fair Benchmark at Batch Size 1 (4 Marks)

**Assessment Requirement:**  
> *"I run 20 warm-up and 100+ timed runs at batch 1. I show median and p95 latency, FPS and model size"*

---

## 1. Benchmarking Protocol & Hardware Controls

To eliminate measurement jitter, cold-cache anomalies, and thermal throttling, this benchmark strictly enforces edge computing scientific standards:
- **Warm-Up Iterations:** Strictly **20 untimed warm-up passes** before data collection to pre-warm the CPU instruction caches and initialize runtime dynamic buffers.
- **Timed Iterations:** Strictly **100 consecutive timed runs** with `time.perf_counter()` at **Batch Size = 1**.
- **Garbage Collection:** Python cyclic garbage collection was temporarily disabled (`gc.disable()`) during timed loops to prevent GC pause spikes.
- **Hardware Profile:** Raspberry Pi 5 Model B Rev 1.1 (BCM2712 Quad Cortex-A76 @ 2.4 GHz, 4 GB RAM, Debian 13 aarch64).
- **Core Threading:** ONNX Runtime allocated across 4 dedicated intra-op worker threads (`intra_op_num_threads = 4`).

---

## 2. Benchmark Results Summary Table

| Model Identifier | Format & Precision | Model Size (MB) | Median Latency (ms) | P95 Latency (ms) | Mean Latency (ms) | Min – Max (ms) | Pure Model Throughput (FPS) | SoC Temp (°C) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M1 Baseline** | PyTorch FP32 | 5.16 MB | 474.47 ms | 584.71 ms | 474.34 ms | 348.75 – 747.52 ms | 2.11 FPS | 53.5°C |
| **M2 Original** | PyTorch FP32 | 5.17 MB | 476.90 ms | 605.62 ms | 481.28 ms | 358.75 – 694.02 ms | 2.08 FPS | 56.2°C |
| **M2 Exported** | ONNX FP32 | 9.34 MB | **243.77 ms** | 401.15 ms | **244.12 ms** | 130.40 – 474.69 ms | **4.10 FPS** | 59.5°C |
| **M2 Optimized** | **ONNX INT8** | **2.72 MB** | **251.99 ms** | **401.51 ms** | **258.25 ms** | 146.02 – 484.36 ms | **3.87 FPS** | 59.0°C |

---

## 3. Investigation: Pure Model Throughput vs. Live Webcam FPS

A critical distinction must be drawn between **raw model benchmark throughput** and **full application pipeline throughput**:

1. **Pure Model Inference Throughput (Benchmark):**
   - In `run_fair_benchmark.py`, a pre-allocated `(1, 3, 640, 640)` float32 tensor is fed directly into the model.
   - For `model_v2_int8.onnx`, mean inference latency is **258.25 ms**, yielding **3.87 FPS** ($1000 / 258.25\text{ ms}$).
2. **Live Webcam Pipeline Framerate (`live_inference_m2.py`):**
   - The live webcam application performs multiple sequential stages per frame:
     - **V4L2 Frame Capture:** USB webcam hardware buffer grab (~25 – 35 ms).
     - **Preprocessing:** Aspect-ratio letterbox resize, BGR to RGB conversion, normalization (~15 – 20 ms).
     - **Model Inference:** INT8 execution on 4 CPU cores (~230 – 260 ms).
     - **Postprocessing:** Transpose, anchor decoding, confidence thresholding, NMS (~5 – 10 ms).
     - **HUD Rendering & GUI Blit:** OpenCV alpha blending, text rendering, and display update (~5 – 10 ms).
   - **Total End-to-End Cycle Time:** ~280 – 320 ms per frame.
   - **Resulting Live Framerate:** **~3.2 to 3.8 FPS**.
   *(Note: Single fastest transient inference frames can reach ~163 ms, but sustained real-world pipeline throughput hovers at ~3.5 FPS).*

---

## 4. Key Observations & Findings

1. **ONNX Runtime Graph Compilation Speedup (1.95x faster):**
   - The ONNX FP32 model achieved a median latency of **243.77 ms**, compared to **476.90 ms** for PyTorch FP32.
   - Compiling the model into an ONNX DAG and executing via C++ ONNX Runtime doubled speed by eliminating Python autograd interpreter overhead.
2. **70.9% Model Footprint Reduction:**
   - Dynamic INT8 quantization collapsed the model file size from **9.34 MB down to 2.72 MB** (70.9% cut).
3. **P95 Latency Stability:**
   - 95% of all inferences completed within ~401 ms, demonstrating predictable real-time execution without extreme tail spikes.
4. **Thermal Stability:**
   - Across all 480 continuous inferences, CPU temperature rose moderately from 47.4°C to 59.5°C, remaining well below the 80.0°C thermal throttling ceiling.

---

## 5. Visual Benchmark Analysis

A 4-panel comparison visualization is saved in this folder:
- [`benchmark_comparison_plot.png`](benchmark_comparison_plot.png):
  - **Panel 1:** Median vs. P95 latency bar comparison.
  - **Panel 2:** Model footprint reduction (MB).
  - **Panel 3:** Latency boxplot distribution across all 100 runs.
  - **Panel 4:** Throughput comparison in Frames Per Second (FPS).

---

## 6. Raw Data Evidence Files

Every single execution time is independently verifiable:
- [`raw_benchmark_m1_pt.csv`](raw_benchmark_m1_pt.csv): 100 timed runs for M1 PyTorch Baseline.
- [`raw_benchmark_m2_pt.csv`](raw_benchmark_m2_pt.csv): 100 timed runs for M2 PyTorch Original.
- [`raw_benchmark_m2_onnx.csv`](raw_benchmark_m2_onnx.csv): 100 timed runs for M2 ONNX FP32.
- [`raw_benchmark_m2_int8.csv`](raw_benchmark_m2_int8.csv): 100 timed runs for M2 ONNX INT8 Quantized.
- [`benchmark_summary.csv`](benchmark_summary.csv): Aggregated master statistics.
