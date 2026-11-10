# Reproducibility Guide: Milestone 2 (Edge AI Optimization)

This document provides exact, deterministic commands to reproduce every result, model export, quantization step, and benchmark in this repository.

---

## 1. Environment & Codebase Setup

```bash
# 1. Clone repository
git clone https://github.com/mithra4325/edge_ai_m2.git
cd edge_ai_m2

# 2. Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install core dependencies
pip install --upgrade pip
pip install torch ultralytics onnx onnxruntime onnxslim opencv-python numpy psutil
```

---

## 2. Model Reproduction Commands

### Step 1: Export Roboflow PyTorch Weights to ONNX
```bash
python3 M2/01_onnx_export/export_onnx.py
```
*Expected Output:* Generates `models/model_v2.onnx` (9.34 MB) with input `[1, 3, 640, 640]` and output `[1, 5, 8400]`.

### Step 2: Convert and Quantize Model to INT8 ONNX
```bash
python3 M2/02_model_conversion_and_live_deployment/quantize_model.py
```
*Expected Output:* Generates `models/model_v2_int8.onnx` (2.72 MB, 70.9% size reduction).

---

## 3. Live Deployment Command

Connect USB webcam to Raspberry Pi 5 USB 3.0/2.0 port:
```bash
bash M2/02_model_conversion_and_live_deployment/run_m2_live.sh
```
*Interactive controls:* `q` = quit, `s` = save frame, `h` = toggle HUD.

---

## 4. Evaluation and Benchmark Commands

### Step 1: Run Accuracy Comparison on Identical Test Set
```bash
python3 M2/03_accuracy_comparison/evaluate_accuracy_comparison.py
```
*Expected Output:*
- Evaluates `model_v1_baseline.pt`, `model_v2_original.pt`, and `model_v2_int8.onnx`.
- Outputs `M2/03_accuracy_comparison/accuracy_comparison.csv`.
- Generates side-by-side comparison images in `M2/03_accuracy_comparison/visual_comparisons/`.

### Step 2: Run Fair 20 Warmup + 100 Timed Runs Benchmark
```bash
python3 M2/04_fair_benchmark/run_fair_benchmark.py
```
*Expected Output:*
- Executes 20 warm-up runs followed by 100 timed runs for all 4 model checkpoints at batch size 1.
- Generates individual raw latency CSV files (`raw_benchmark_*.csv`).
- Generates master summary CSV `M2/04_fair_benchmark/benchmark_summary.csv`.
