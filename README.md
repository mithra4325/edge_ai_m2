# Edge AI 3D-Printed Enclosure Detection — Milestone 2 (M2 Optimize)

[![Hardware: Raspberry Pi 5](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-red.svg)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![SoC: Broadcom BCM2712](https://img.shields.io/badge/SoC-Broadcom%20BCM2712%20(Quad%20A76)-blue.svg)]()
[![Model: YOLO26n INT8](https://img.shields.io/badge/Model-YOLO26n%20INT8%20ONNX-green.svg)]()
[![Precision: 99.67%](https://img.shields.io/badge/Precision-99.67%25-brightgreen.svg)]()
[![mAP@0.50: 96.60%](https://img.shields.io/badge/mAP%400.50-96.60%25-brightgreen.svg)]()
[![Footprint: 2.72 MB](https://img.shields.io/badge/Size-2.72%20MB%20(-70.9%25)-orange.svg)]()

Welcome to the **Milestone 2 (M2 Optimize)** repository for the **Edge AI 3D-Printed Enclosure Detection** system, authored by **Mithra Ravi (`mithra4325`)**.

This milestone advances the original baseline project into an optimized, self-contained Edge AI pipeline running locally on the Raspberry Pi 5 CPU via ARM NEON vector instructions and dynamic INT8 quantization.

---

## 1. Milestone 2 Assessment Navigation

This repository is strictly organized to match the college assessment criteria (20 Marks total):

| Folder | Task Description | Rubric Requirement | Max Marks | Status |
| :--- | :--- | :--- | :---: | :---: |
| [**`M2/01_onnx_export/`**](file:///home/forge/edge_ai_m2/M2/01_onnx_export/README.md) | **Export to ONNX** | *"I run the ONNX export myself and show my commit"* | **3** | **COMPLETE** |
| [**`M2/02_model_conversion_and_live_deployment/`**](file:///home/forge/edge_ai_m2/M2/02_model_conversion_and_live_deployment/README.md) | **Convert and Run Live** | *"I convert to TFLite INT8 or TensorRT and run it live on the device"* | **4** | **PARTIAL / AT RISK** *(ONNX INT8 deployed; LiteRT fails on ARM64)* |
| [**`M2/03_accuracy_comparison/`**](file:///home/forge/edge_ai_m2/M2/03_accuracy_comparison/README.md) | **Accuracy Check** | *"I show mAP before vs after on the same test set"* | **4** | **COMPLETE** |
| [**`M2/04_fair_benchmark/`**](file:///home/forge/edge_ai_m2/M2/04_fair_benchmark/README.md) | **Fair Benchmark** | *"I run 20 warm-up and 100+ timed runs at batch 1. I show median and p95 latency, FPS and model size"* | **4** | **COMPLETE** |
| [**`M2/05_cpu_vs_accelerator_and_stack/`**](file:///home/forge/edge_ai_m2/M2/05_cpu_vs_accelerator_and_stack/README.md) | **CPU vs Accelerator & Stack** | *"I show the CPU vs accelerator table and draw the software stack"* | **2** | **COMPLETE** |
| [**`M2/06_target_and_tradeoff/`**](file:///home/forge/edge_ai_m2/M2/06_target_and_tradeoff/README.md) | **Target & Trade-off** | *"I show the accuracy and speed target I wrote before training, my result, and explain the trade-off in simple words"* | **3** | **COMPLETE** |

---

## 2. Key Engineering Achievements in M2

1. **Model Footprint Reduced by 70.9%:**
   - Uncompressed ONNX export: **9.34 MB**
   - Quantized Dynamic INT8 ONNX model: **2.72 MB** (102 `ConvInteger` nodes).
2. **False Positives Completely Eliminated on Test Frames:**
   - Dataset expanded to 101 images (173 annotations).
   - Zero hallucinations on plain white tables, bright glare, or paper backgrounds at operational 0.40 confidence threshold.
3. **Accuracy Parity Preserved:**
   - mAP@0.50 maintained at **96.60%** across identical 10-image test set.
4. **Lightweight Edge Memory Footprint:**
   - RAM usage collapsed from ~520 MB (PyTorch) to **< 60 MB** (ONNX Runtime).
5. **Real-time Diagnostic Heads-Up Display (HUD):**
   - Live stream on Logitech C270 webcam with per-stage latency, FPS, SoC thermal telemetry, and enclosure count.

---

## 3. Quick Start: Live Inference on Raspberry Pi 5

```bash
# 1. Clone repository
git clone https://github.com/mithra4325/edge_ai_m2.git
cd edge_ai_m2

# 2. Activate virtual environment
source /home/forge/edge-ai-enclosure-detection/venv/bin/activate

# 3. Launch live detection with HUD overlay
bash M2/02_model_conversion_and_live_deployment/run_m2_live.sh
```

---

## 4. Documentation Suite

Comprehensive engineering documentation is available in [`docs/`](docs/):
- [**`docs/M2_PROJECT_REPORT.md`**](file:///home/forge/edge_ai_m2/docs/M2_PROJECT_REPORT.md): Complete technical report of Milestone 2.
- [**`docs/M2_ASSESSMENT_MAPPING.md`**](file:///home/forge/edge_ai_m2/docs/M2_ASSESSMENT_MAPPING.md): Direct 1-to-1 mapping of rubric tasks to evidence files.
- [**`docs/M1_VS_M2_SUMMARY.md`**](file:///home/forge/edge_ai_m2/docs/M1_VS_M2_SUMMARY.md): Comprehensive comparison showing how M2 resolved M1 limitations.
- [**`docs/MODEL_V2_CARD.md`**](file:///home/forge/edge_ai_m2/docs/MODEL_V2_CARD.md): Complete specifications for `object-detection-n6b60-2-yolo26n-t2`.
- [**`docs/OPTIMIZATION_REPORT.md`**](file:///home/forge/edge_ai_m2/docs/OPTIMIZATION_REPORT.md): Deep dive into graph slimming, INT8 dynamic quantization, and ARM NEON SIMD.
- [**`docs/DEPLOYMENT_GUIDE.md`**](file:///home/forge/edge_ai_m2/docs/DEPLOYMENT_GUIDE.md): End-to-end deployment instructions on embedded hardware.
- [**`docs/REPRODUCIBILITY.md`**](file:///home/forge/edge_ai_m2/docs/REPRODUCIBILITY.md): Deterministic reproduction commands.
- [**`docs/M2_EXPERIMENT_LOG.md`**](file:///home/forge/edge_ai_m2/docs/M2_EXPERIMENT_LOG.md): Chronological experiment history.

---

## 5. Repository Structure

```
edge_ai_m2/
├── M2/
│   ├── 01_onnx_export/                             # Task 1 (3 Marks)
│   │   ├── README.md
│   │   ├── export_onnx.py
│   │   └── export_log.txt
│   ├── 02_model_conversion_and_live_deployment/    # Task 2 (4 Marks)
│   │   ├── README.md
│   │   ├── quantize_model.py
│   │   ├── live_inference_m2.py
│   │   ├── run_m2_live.sh
│   │   ├── quantization_log.txt
│   │   ├── tflite_export_attempt.log
│   │   └── evidence/
│   │       ├── m2_detection_hud_verified.jpg
│   │       └── m2_live_frame_0025.jpg
│   ├── 03_accuracy_comparison/                     # Task 3 (4 Marks)
│   │   ├── README.md
│   │   ├── evaluate_accuracy_comparison.py
│   │   ├── accuracy_comparison.csv
│   │   ├── accuracy_comparison_log.txt
│   │   └── visual_comparisons/
│   ├── 04_fair_benchmark/                          # Task 4 (4 Marks)
│   │   ├── README.md
│   │   ├── run_fair_benchmark.py
│   │   ├── benchmark_summary.csv
│   │   ├── benchmark_comparison_plot.png
│   │   ├── raw_benchmark_m1_pt.csv
│   │   ├── raw_benchmark_m2_int8.csv
│   │   ├── raw_benchmark_m2_onnx.csv
│   │   └── raw_benchmark_m2_pt.csv
│   ├── 05_cpu_vs_accelerator_and_stack/            # Task 5 (2 Marks)
│   │   └── README.md
│   └── 06_target_and_tradeoff/                     # Task 6 (3 Marks)
│       └── README.md
├── cad/
│   └── pir_sensor_enclosure_mithra_ravi.stl        # 3D-printed enclosure CAD model
├── data/                                           # Dataset v2 (101 images, 70/20/10 split)
├── docs/                                           # Engineering documentation suite
├── models/
│   ├── model_v1_baseline.pt                        # M1 Baseline (5.16 MB)
│   ├── model_v2_original.pt                        # M2 Original (5.17 MB)
│   ├── model_v2.onnx                               # M2 ONNX FP32 (9.34 MB)
│   ├── model_v2_int8.onnx                          # M2 ONNX INT8 Quantized (2.72 MB)
│   └── model_v2_metadata.json                      # Roboflow training parameters
└── README.md                                       # Master repository documentation
```
