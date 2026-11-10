# M2.2: Model Conversion and Live Edge Deployment (4 Marks)

**Assessment Requirement:**  
> *"I convert to TFLite INT8 or TensorRT and run it live on the device"*

---

## 1. Hardware and Framework Target Analysis

When deploying an Edge AI object detector on embedded hardware, the runtime execution engine must align with the physical SoC architecture:

### Why TensorRT is Inapplicable:
- **TensorRT** is NVIDIA's proprietary closed-source inference optimizer and runtime.
- It requires an **NVIDIA GPU with CUDA Tensor Cores** (such as NVIDIA Jetson Nano, Orin Nano, Xavier, or desktop RTX GPUs).
- The Raspberry Pi 5 uses a **Broadcom BCM2712 SoC** featuring a Quad-core ARM Cortex-A76 CPU and a Broadcom VideoCore VII GPU (which lacks CUDA support).
- Therefore, TensorRT cannot physically execute on a Raspberry Pi 5.

### Runtime Selection on Debian 13 (Trixie) 64-bit / Python 3.13:
- Raspberry Pi OS / Debian 13 ships with **Python 3.13.5**.
- Upstream TensorFlow / TFLite runtime wheels for Linux `aarch64` are currently pre-built only up to Python 3.12.
- **ONNX Runtime (v1.31.1)** is fully compiled with native **ARM NEON vector SIMD extensions** for Python 3.13 on `aarch64`.
- ONNX Runtime Dynamic INT8 quantization provides the exact mathematical quantization (8-bit signed integer weights, dynamic activation scaling) required for embedded edge deployment, achieving a **70.9% reduction in model size** while running live on the CPU.

---

## 2. Quantization Results Summary

| Metric | Source Model (`model_v2.onnx`) | Quantized Model (`model_v2_int8.onnx`) | Impact / Benefit |
| :--- | :--- | :--- | :--- |
| **Precision** | Float32 (32-bit floating point) | **INT8 (8-bit signed integer)** | 4x lower memory bandwidth per weight |
| **Model Size** | **9.34 MB** | **2.72 MB** | **70.9% footprint reduction** |
| **Execution Engine** | ONNX Runtime CPU | **ONNX Runtime ARM64 NEON INT8** | Low-latency edge execution |
| **Sample Confidence** | 96.77% | **96.83%** | **Parity preserved (< 0.1% delta)** |
| **Quantization Type** | None | Dynamic Quantization (`QInt8`) | Fast conversion, zero calibration drift |

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

### Conversion Pipeline Details:
- Inspects input ONNX model `models/model_v2.onnx` (9.34 MB).
- Applies `onnxruntime.quantization.quantize_dynamic` targeting `Conv`, `MatMul`, and `Gemm` operators.
- Generates `models/model_v2_int8.onnx` (2.72 MB).
- Instantiates an `InferenceSession` with `CPUExecutionProvider` to verify graph input (`images: [1, 3, 640, 640]`) and output (`output0: [1, 5, 8400]`).
- Full execution log is preserved in [`quantization_log.txt`](quantization_log.txt).

---

## 4. How to Launch Live Detection on Raspberry Pi 5

We provide a one-click launcher shell script as well as direct Python invocation:

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

## 5. Live Pipeline Architecture and HUD Features

The live detection pipeline (`live_inference_m2.py`) incorporates real-time edge diagnostics:
1. **Camera Frame Capture:** Grabs live 640x480 video frames from the Logitech C270 USB webcam on `/dev/video0`.
2. **Letterbox Preprocessing:** Scales frames to 640x640 maintaining aspect ratio with symmetric gray padding.
3. **Quantized Inference:** Feeds the normalized RGB tensor to the 2.72 MB INT8 model.
4. **NMS Postprocessing:** Applies Non-Maximum Suppression at IoU 0.45 and Confidence 0.40.
5. **HUD Diagnostic Overlay:**
   - **Bounding Box & Label:** High-contrast neon tag showing class name (`ENCLOSURE`) and confidence percentage.
   - **FPS Counter:** Real-time rolling average framerate.
   - **Latency Telemetry:** Inference latency and total pipeline round-trip time in milliseconds.
   - **Hardware Telemetry:** Direct SoC thermal readout from `/sys/class/thermal/thermal_zone0/temp`.
   - **Object Counter:** Real-time tally of detected 3D-printed enclosures.

---

## 6. Real Device Evidence

The live detection pipeline was executed on the physical Raspberry Pi 5. Generated evidence frames are stored in [`evidence/`](evidence/):
- [`m2_detection_hud_verified.jpg`](evidence/m2_detection_hud_verified.jpg): Verified HUD output showing detected enclosure with 96.3% confidence, latency HUD, and FPS telemetry.
- [`m2_live_frame_0025.jpg`](evidence/m2_live_frame_0025.jpg): Physical webcam capture during live device stream validation.

---

## 7. Student Demonstration Checklist

When presenting this task to the examiner:
1. Show `models/model_v2_int8.onnx` file in terminal (`ls -lh models/model_v2_int8.onnx`) showing **2.72 MB**.
2. Run `python3 M2/02_model_conversion_and_live_deployment/quantize_model.py` to demonstrate dynamic INT8 quantization.
3. Run `bash M2/02_model_conversion_and_live_deployment/run_m2_live.sh` to launch live detection on the Raspberry Pi 5 desktop.
4. Point out the live HUD overlay showing latency, framerate, CPU temperature, and enclosure detection bounding boxes.
5. Explain clearly why TensorRT is NVIDIA-only and why ONNX Runtime INT8 on ARM NEON is the optimal edge deployment choice for Raspberry Pi 5.
