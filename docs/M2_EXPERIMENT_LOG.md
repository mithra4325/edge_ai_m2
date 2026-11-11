# M2 Chronological Experiment Log

This log documents all experimental trials, model versions, conversions, and diagnostic runs performed during Milestone 2.

---

### Experiment 1: Model Retrieval and Verification
- **Date & Time:** 2026-11-10 19:48 IST
- **Target Model:** `object-detection-n6b60-2-yolo26n-t2`
- **Method:** Roboflow Python SDK signed S3 endpoint retrieval via project API key.
- **Downloaded Weight:** `models/model_v2_original.pt` (5.17 MB, MD5: `ed26737f31d3f12d63cf3d27e9a4515c`).
- **Validation:** Inspected with Ultralytics; verified 120 layers, 2,375,031 parameters, 5.3 GFLOPs, 1 class (`enclosure`).

### Experiment 2: Baseline Model Isolation
- **Date & Time:** 2026-11-10 19:49 IST
- **Action:** Read-only preservation of M1 model from `/home/forge/edge-ai-enclosure-detection/models/best.pt` to `/home/forge/edge_ai_m2/models/model_v1_baseline.pt` (5.16 MB, MD5: `b2e2066e20ac81a0ec7346debc1ec52f`).
- **Verification:** Both M1 and M2 weights verified intact with distinct MD5 checksums.

### Experiment 3: Full ONNX Export and Graph Slimming
- **Date & Time:** 2026-11-10 19:51 IST
- **Command:** `python3 M2/01_onnx_export/export_onnx.py`
- **Parameters:** `opset=17`, `imgsz=640`, `simplify=True` (`onnxslim 0.1.99`).
- **Result:** Successfully created `models/model_v2.onnx` (9.34 MB).
- **Parity Check:** Numerical parity verified on `IMG_9716_HEIC...` (PyTorch conf: 0.9677, ONNX conf: 0.9677).

### Experiment 4: Dynamic INT8 Quantization
- **Date & Time:** 2026-11-10 19:54 IST
- **Command:** `python3 M2/02_model_conversion_and_live_deployment/quantize_model.py`
- **Method:** `onnxruntime.quantization.quantize_dynamic` targeting `Conv`, `MatMul`, `Gemm` operators.
- **Result:** Generated `models/model_v2_int8.onnx` (2.72 MB, 102 `ConvInteger` nodes).
- **Footprint Impact:** 70.9% size reduction. Verified inference on `IMG_9716_HEIC...` (Conf: 0.9683).

### Experiment 5: LiteRT / TFLite Export Investigation
- **Date & Time:** 2026-11-11 17:56 IST
- **Command:** `m.export(format='tflite', int8=True)`
- **Finding:** Hard assertion failure: `AssertionError: LiteRT export only supported on Linux x86 and macOS`.
- **Evidence:** Log saved to `M2/02_model_conversion_and_live_deployment/tflite_export_attempt.log`. Proves TFLite cannot build on Linux ARM64.

### Experiment 6: Live Webcam Deployment & GUI HighGUI Fix
- **Date & Time:** 2026-11-11 17:57 IST
- **Device:** Logitech C270 USB Webcam on `/dev/video0`.
- **Finding:** `opencv-python-headless` in venv shadowed HighGUI. Removed `opencv-python-headless` so OpenCV uses native GTK display.
- **Result:** `cv2.imshow` successfully tested on `DISPLAY=:0`.

### Experiment 7: Official Test Set Evaluation Across Full PR Curve
- **Date & Time:** 2026-11-11 17:58 IST
- **Command:** `python3 M2/03_accuracy_comparison/evaluate_accuracy_comparison.py`
- **Dataset:** 10 physical test images (16 enclosure ground truth instances).
- **Official Results:**
  - M1 Baseline: Precision 93.90%, Recall 100.00%, mAP@0.50 99.15%, mAP@0.5:0.95 86.92%.
  - M2 Original: Precision 99.66%, Recall 93.75%, mAP@0.50 96.03%, mAP@0.5:0.95 82.57%.
  - M2 INT8 ONNX: Precision 99.67%, Recall 93.75%, mAP@0.50 96.60%, mAP@0.5:0.95 80.10%.
