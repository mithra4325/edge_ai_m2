#!/usr/bin/env python3
"""
M2.4 Fair Benchmark Script (Raspberry Pi 5)
Executes strictly 20 warm-up iterations followed by 100 timed iterations at Batch Size 1.
Benchmarks:
  1. M1 Baseline PyTorch FP32 (model_v1_baseline.pt)
  2. M2 Original PyTorch FP32 (model_v2_original.pt)
  3. M2 Exported ONNX FP32 (model_v2.onnx)
  4. M2 Quantized ONNX INT8 (model_v2_int8.onnx)
Computes Median Latency, P95 Latency, Mean Latency, Min, Max, FPS, and Model Size.
Saves raw per-iteration data to CSV and generates comparison summary.
"""

import os
import sys
import time
import csv
import gc
import psutil
import numpy as np
import torch
import onnxruntime as ort

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 0.0

from ultralytics import YOLO

def benchmark_pytorch(model_path, num_warmup=20, num_timed=100):
    print(f"\n[*] Benchmarking PyTorch Model: {os.path.basename(model_path)}")
    yolo_obj = YOLO(model_path)
    model = yolo_obj.model
    model.eval()

    dummy_input = torch.zeros((1, 3, 640, 640), dtype=torch.float32)

    # 20 Warmup runs
    print(f"    - Running {num_warmup} warm-up iterations (batch=1)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # 100 Timed runs
    print(f"    - Running {num_timed} timed iterations (batch=1)...")
    latencies = []
    gc.disable()
    with torch.no_grad():
        for _ in range(num_timed):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
    gc.enable()

    del model
    gc.collect()
    return latencies

def benchmark_onnx(model_path, num_warmup=20, num_timed=100):
    print(f"\n[*] Benchmarking ONNX Model: {os.path.basename(model_path)}")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_opts, providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)

    # 20 Warmup runs
    print(f"    - Running {num_warmup} warm-up iterations (batch=1)...")
    for _ in range(num_warmup):
        _ = session.run(None, {inp_name: dummy_input})

    # 100 Timed runs
    print(f"    - Running {num_timed} timed iterations (batch=1)...")
    latencies = []
    gc.disable()
    for _ in range(num_timed):
        t0 = time.perf_counter()
        _ = session.run(None, {inp_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
    gc.enable()

    del session
    gc.collect()
    return latencies

def compute_stats(latencies):
    arr = np.array(latencies)
    mean_lat = float(np.mean(arr))
    median_lat = float(np.median(arr))
    p95_lat = float(np.percentile(arr, 95))
    p99_lat = float(np.percentile(arr, 99))
    min_lat = float(np.min(arr))
    max_lat = float(np.max(arr))
    std_lat = float(np.std(arr))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    return {
        "mean": mean_lat,
        "median": median_lat,
        "p95": p95_lat,
        "p99": p99_lat,
        "min": min_lat,
        "max": max_lat,
        "std": std_lat,
        "fps": fps
    }

def main():
    print("=" * 70)
    print(" M2 TASK 4: FAIR BENCHMARK (20 WARM-UP + 100 TIMED RUNS @ BATCH 1)")
    print("=" * 70)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    out_dir = os.path.join(repo_root, "M2", "04_fair_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    models_info = [
        {"name": "M1 Baseline (PyTorch FP32)", "path": "models/model_v1_baseline.pt", "type": "pytorch", "tag": "m1_pt"},
        {"name": "M2 Original (PyTorch FP32)", "path": "models/model_v2_original.pt", "type": "pytorch", "tag": "m2_pt"},
        {"name": "M2 Exported (ONNX FP32)", "path": "models/model_v2.onnx", "type": "onnx", "tag": "m2_onnx"},
        {"name": "M2 Optimized (ONNX INT8)", "path": "models/model_v2_int8.onnx", "type": "onnx", "tag": "m2_int8"}
    ]

    summary_rows = []
    num_warmup = 20
    num_timed = 100

    temp_start = get_cpu_temp()
    print(f"[*] Initial CPU Temperature: {temp_start:.1f}°C")
    print(f"[*] Available CPU Cores:      {psutil.cpu_count(logical=True)}")
    print(f"[*] System Total RAM:         {psutil.virtual_memory().total / (1024**3):.2f} GB")

    for item in models_info:
        full_path = os.path.join(repo_root, item["path"])
        size_mb = os.path.getsize(full_path) / (1024 * 1024)

        if item["type"] == "pytorch":
            lats = benchmark_pytorch(full_path, num_warmup, num_timed)
        else:
            lats = benchmark_onnx(full_path, num_warmup, num_timed)

        stats = compute_stats(lats)
        cur_temp = get_cpu_temp()

        # Save raw latencies CSV
        raw_csv_path = os.path.join(out_dir, f"raw_benchmark_{item['tag']}.csv")
        with open(raw_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Iteration", "Latency_ms"])
            for idx, val in enumerate(lats, start=1):
                w.writerow([idx, f"{val:.3f}"])

        print(f"    [+] Saved raw data to: raw_benchmark_{item['tag']}.csv")
        print(f"    [+] Stats: Median={stats['median']:.2f} ms | P95={stats['p95']:.2f} ms | Mean={stats['mean']:.2f} ms | FPS={stats['fps']:.2f} | Size={size_mb:.2f} MB")

        summary_rows.append({
            "model_name": item["name"],
            "format": item["type"].upper(),
            "size_mb": f"{size_mb:.2f}",
            "median_ms": f"{stats['median']:.2f}",
            "p95_ms": f"{stats['p95']:.2f}",
            "mean_ms": f"{stats['mean']:.2f}",
            "min_ms": f"{stats['min']:.2f}",
            "max_ms": f"{stats['max']:.2f}",
            "fps": f"{stats['fps']:.2f}",
            "temp_c": f"{cur_temp:.1f}"
        })

    # Write summary CSV
    summary_csv = os.path.join(out_dir, "benchmark_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "format", "size_mb", "median_ms", "p95_ms", "mean_ms", "min_ms", "max_ms", "fps", "temp_c"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[+] Master benchmark summary written to: {summary_csv}")
    print("=" * 70)
    print(" BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
