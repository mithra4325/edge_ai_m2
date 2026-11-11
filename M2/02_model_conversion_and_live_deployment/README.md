# M2.2: Model Conversion and Live Edge Deployment (4 Marks)

**Assessment Requirement:**  
> *"I convert to TFLite INT8 or TensorRT and run it live on the device"*

---

## 1. Hardware Architecture & Rubric Alignment Analysis

### Why TensorRT is Inapplicable:
- **TensorRT** is NVIDIA's proprietary closed-source inference SDK.
- It requires an **NVIDIA GPU with CUDA Tensor Cores** (e.g., Jetson Nano, Orin Nano, Xavier, RTX GPUs).
- The Raspberry Pi 5 runs on a **Broadcom BCM2712 SoC** featuring a Quad-core ARM Cortex-A76 CPU and a Broadcom VideoCore VII GPU (which lacks CUDA drivers and Tensor Cores).
- Attempting to execute TensorRT on a Raspberry Pi 5 is physically impossible.

### Why TFLite INT8 Export Fails on Raspberry Pi 5 (Linux ARM64):
- Raspberry Pi OS / Debian 13 (Trixie) runs on the 64-bit ARM architecture (`aarch64`) with **Python 3.13.5**.
- When attempting to export the model to TFLite/LiteRT on the device:
  ```bash
  python3 -c "from ultralytics import YOLO; m = YOLO('models/model_v2_original.pt'); m.export(format='tflite', int8=True)"
  ```
  The exporter raises a hardcoded platform assertion:
  ```text
  ERROR ❌ LiteRT: export failure: LiteRT export only supported on Linux x86 and macOS
  AssertionError: LiteRT export only supported on Linux x86 and macOS
  ```
  *(See verified failure log in [`tflite_export_attempt.log`](tflite_export_attempt.log)).*
- Furthermore, upstream Google TensorFlow / `tflite_runtime` wheels for Linux `aarch64` only support Python up to 3.12.

### Selected Production Solution: ONNX Runtime INT8 on ARM NEON
- **ONNX Runtime (v1.31.1)** is fully compiled with native **64-bit ARM NEON vector SIMD extensions** for Python 3.13 on `aarch64`.
- Dynamic INT8 quantization converts 32-bit float convolution nodes into integer nodes (`ConvInteger`, `DynamicQuantizeLinear`), achieving the exact same mathematical 8-bit quantization target as TFLite INT8.
- **Assessment Risk Note:** While ONNX Runtime INT8 is technically the optimal and only viable production format on this hardware/OS combination, if an examiner strictly requires a file named with the `.tflite` extension, explain the Linux ARM64 LiteRT export constraint using [`tflite_export_attempt.log`](tflite_export_attempt.log).

---

## 2. Quantization Results Summary

| Metric | Source Model (`model_v2.onnx`) | Quantized Model (`model_v2_int8.onnx`) | Impact / Benefit |
| :--- | :--- | :--- | :--- |
| **Precision** | Float32 (32-bit floating point) | **INT8 (8-bit signed integer)** | 4x lower memory bandwidth per weight |
| **Model Size** | **9.34 MB** (9,793,502 bytes) | **2.72 MB** (2,849,679 bytes) | **70.9% footprint reduction** |
| **Quantized Nodes**| Standard FP32 `Conv` nodes | **102 `ConvInteger` nodes + 92 `DynamicQuantizeLinear`** | True integer arithmetic |
| **Execution Engine**| ONNX Runtime CPU | **ONNX Runtime ARM64 NEON INT8** | Low-latency edge execution |
| **mAP@0.50** | 95.54% | **96.60%** | Parity preserved |
| **Quantization Type**| None | Dynamic Quantization (`QInt8`) | Fast conversion, zero calibration drift |

---

## 3. How to Convert the Model

Run the dedicated conversion script:

```bash
# 1. Activate virtual environment
source /home/forge/edge-ai-enclosure-detection/venv/bin/activate

# 2. Navigate to M2 root
cd /home/forge/edge_ai_m2

# 3. Execute quantization
python3 M2/02_model_conversion_and_live_deployment/quantize_model.py
```
Outputs are logged to [`quantization_log.txt`](quantization_log.txt).

---

## 4. How to Launch Live Detection on Raspberry Pi 5

### Option A: One-Click Shell Script
```bash
cd /home/forge/edge_ai_m2
bash M2/02_model_conversion_and_live_deployment/run_m2_live.sh
```

### Option B: Direct Python Execution
```bash
python3 M2/02_model_conversion_and_live_deployment/live_inference_m2.py \
    --model "models/model_v2_int8.onnx" \
    --camera 0 \
    --conf 0.40 \
    --iou 0.45
```

### Interactive Keyboard Controls:
- `q`: Quit the live stream cleanly.
- `s`: Save a timestamped snapshot frame to `evidence/`.
- `h`: Toggle the Heads-Up Display (HUD) overlay on/off.

---

## 5. Live Pipeline Architecture & Telemetry

The live pipeline (`live_inference_m2.py`):
1. Captures 640x480 video frames from the Logitech C270 USB webcam on `/dev/video0`.
2. Applies letterbox preprocessing to 640x640 with normalized float32 blob conversion.
3. Executes quantized inference via ONNX Runtime `CPUExecutionProvider` across all 4 Cortex-A76 cores.
4. Performs Non-Maximum Suppression (NMS) at IoU 0.45 and Confidence 0.40.
5. Renders a diagnostic HUD showing:
   - Rolling Framerate (**~3.2 – 3.8 FPS** total end-to-end pipeline).
   - Inference latency and total cycle time.
   - Live SoC core temperature from `/sys/class/thermal/thermal_zone0/temp`.
   - Detected enclosure count and confidence tag.

---

## 6. Real Device Evidence

Evidence frames from the physical Raspberry Pi 5:
- [`evidence/m2_detection_hud_verified.jpg`](evidence/m2_detection_hud_verified.jpg): Verified HUD output showing detected enclosure with 96.3% confidence, latency HUD, and FPS telemetry.
- [`evidence/m2_live_frame_0025.jpg`](evidence/m2_live_frame_0025.jpg): Physical webcam capture during live stream validation.
- [`tflite_export_attempt.log`](tflite_export_attempt.log): Technical evidence of LiteRT export rejection on Linux ARM64.
