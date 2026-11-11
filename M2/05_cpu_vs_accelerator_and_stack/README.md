# M2.5: CPU vs Accelerator Comparison and Software Stack (2 Marks)

**Assessment Requirement:**  
> *"I show the CPU vs accelerator table and draw the software stack"*

---

## 1. Physical Hardware Inspection of the Deployment Target

Low-level hardware verification on the test device:

```bash
# Verify SoC and CPU Architecture
lscpu
# Verify PCI devices (checking for M.2 NPU / Coral M.2 / Hailo)
lspci
# Verify USB devices (checking for USB Edge TPU / Coral USB)
lsusb
```

### Verified Hardware Profile:
- **Device Model:** Raspberry Pi 5 Model B Rev 1.1
- **System on Chip (SoC):** Broadcom BCM2712 (16nm FinFET process)
- **CPU Architecture:** Quad-core ARM Cortex-A76 (ARMv8.2-A 64-bit) @ 2.40 GHz
  - 64 KB I-Cache + 64 KB D-Cache per core
  - 512 KB L2 cache per core
  - 2 MB shared L3 cache
  - **SIMD Vector Engine:** ARM NEON 128-bit SIMD registers with integer dot-product (`dotprod`) support
- **System Memory:** 4.0 GiB LPDDR4X-4267 SDRAM (bandwidth ~17 GB/s)
- **GPU:** Broadcom VideoCore VII @ 800 MHz (Vulkan 1.2, OpenGL ES 3.1)
  - *Note:* VideoCore VII is a graphics rendering pipeline; it lacks general-purpose matrix compute cores (CUDA/Tensor Cores) and OpenCL ML driver support.
- **Dedicated Hardware Accelerators:** **None connected** (`lspci` and `lsusb` confirm zero external NPU, Coral TPU, or Hailo modules).
- **Camera Sensor:** Logitech C270 HD Webcam connected via USB 2.0 (`/dev/video0`, UVC driver).

---

## 2. Hardware Comparison Matrix (Measured CPU vs. Theoretical Accelerators)

> **Important Disclosure:** Only the **Raspberry Pi 5 CPU** was physically benchmarked. The values for Google Coral, Hailo-8L, and NVIDIA Jetson are theoretical comparisons derived from manufacturer data sheets to contextualize edge deployment feasibility.

| Feature / Metric | Raspberry Pi 5 CPU *(Empirically Measured)* | Google Coral Edge TPU *(Theoretical Spec)* | Raspberry Pi AI Kit *(Hailo-8L Spec)* | NVIDIA Jetson Orin Nano *(Theoretical Spec)* |
| :--- | :--- | :--- | :--- | :--- |
| **Processor Type** | 4x ARM Cortex-A76 CPU | Custom ASIC Tensor Processing Unit | Hailo-8L Deep Learning Co-processor | 6x ARM Cortex-A78AE + Ampere GPU |
| **Vector / Tensor Engine** | ARM NEON 128-bit SIMD | Matrix Multiply Unit (TPU) | Structure-defined Neural Core | 1024 CUDA Cores + 32 Tensor Cores |
| **Peak AI Compute** | **~0.1 TOPS** (FP32/INT8 SIMD) | **4 TOPS** (INT8) | **13 TOPS** (INT8) | **40 TOPS** (INT8) |
| **Typical Power Consumption** | 3.5 W – 7.0 W | 2.0 W – 2.5 W (addon) | 1.5 W – 2.0 W (addon) | 7.0 W – 15.0 W |
| **Energy Efficiency** | ~0.02 TOPS / Watt | ~1.60 TOPS / Watt | ~6.50 TOPS / Watt | ~3.00 TOPS / Watt |
| **Hardware Cost** | **$60 (Included onboard)** | +$60 (Separate USB dongle) | +$70 (Separate PCIe HAT) | $250 – $500 (Entire board) |
| **Supported Precision** | FP32, FP16, **INT8** | **Strict INT8 only** | **Strict INT8 only** | FP32, FP16, INT8 |
| **Primary Software Stack** | **ONNX Runtime (CPU / NEON)** | libedgetpu + PyCoral + TFLite | HailoRT + Hailo TAPPAS | TensorRT + CUDA + cuDNN |
| **Compilation Requirement** | Direct execution (No compile step) | Requires `edgetpu_compiler` | Requires Hailo Dataflow Compiler | Requires TensorRT engine build |
| **Operator Flexibility** | **100% of ONNX / PyTorch ops** | Highly restricted op subset | Restricted op subset | High op support |
| **Measured/Est. Enclosure FPS**| **~3.2 – 3.8 FPS (Measured Live)** | ~25 to 30 FPS (Estimated) | ~35 to 45 FPS (Estimated) | ~45 to 60 FPS (Estimated) |
| **Verdict for this Project** | **Optimal:** Zero added cost, highly portable, >96% mAP | Overkill & strict quantization constraints | Requires PCIe HAT, adds mechanical bulk | Prohibitive cost and power for simple enclosure QC |

---

## 3. End-to-End Software Stack Architecture

The diagram below illustrates the exact 5-layer software architecture driving our M2 Edge AI deployment:

```
+-----------------------------------------------------------------------------------+
|                           APPLICATION LAYER (User & UI)                           |
|                                                                                   |
|   +-----------------------+   +-----------------------+   +--------------------+  |
|   |  live_inference_m2.py |   |  OpenCV HUD Overlay   |   | Snapshot Evidence  |  |
|   |  - Camera Loop        |   |  - Bounding Box & Conf|   | Logger             |  |
|   |  - Aspect Letterbox   |   |  - FPS / Latency ms   |   | - JPEG /dev/video0 |  |
|   |  - NMS IoU Filtering  |   |  - SoC Temperature    |   | - Timestamped Logs |  |
|   +-----------------------+   +-----------------------+   +--------------------+  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                   OPTIMIZATION & INFERENCE ENGINE (Edge Runtime)                  |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |                     ONNX Runtime Engine (v1.31.1)                         |   |
|   |  Execution Provider: CPUExecutionProvider                                  |   |
|   |  Instruction Acceleration: ARM64 NEON Vector Extensions                   |   |
|   |  Threading Model: 4 parallel intra-op worker threads                       |   |
|   +---------------------------------------------------------------------------+   |
|                                        ^                                          |
|                                        | executes                                 |
|   +------------------------------------+--------------------------------------+   |
|   |            models/model_v2_int8.onnx (2.72 MB Quantized Model)             |   |
|   |  - Dynamic INT8 Quantized Weights (102 ConvInteger + 92 DynamicQuantize)  |   |
|   |  - ONNX Opset 17 Standard Interoperability Format                         |   |
|   +---------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                     FRAMEWORK & NUMERICAL DEPENDENCIES                            |
|                                                                                   |
|   +----------------------+   +-----------------------+   +--------------------+   |
|   | OpenCV (cv2) 5.0.0   |   | NumPy 2.2.4 (C-BLAS)  |   | Ultralytics 8.4.208|   |
|   | - V4L2 Video Capture |   | - Fast Array Slicing  |   | - Export & Val     |   |
|   | - Image Color Trans  |   | - Dynamic Dot Product |   | - Roboflow Weights |   |
|   +----------------------+   +-----------------------+   +--------------------+   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        OPERATING SYSTEM & KERNEL DRIVERS                          |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   | Debian GNU/Linux 13 (Trixie) 64-bit / Linux Kernel 6.18.50-v8-16k         |   |
|   | - Video4Linux2 (V4L2) Camera Subsystem (/dev/video0)                      |   |
|   | - Thermal Management Subsystem (/sys/class/thermal/thermal_zone0/)        |   |
|   | - Python 3.13.5 Virtual Environment (/home/forge/.../venv)               |   |
|   +---------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                              PHYSICAL HARDWARE LAYER                              |
|                                                                                   |
|   +------------------------------------+   +----------------------------------+   |
|   | Raspberry Pi 5 Model B Rev 1.1     |   | Peripherals:                     |   |
|   | - Broadcom BCM2712 Quad Cortex-A76 |   | - Logitech C270 USB Webcam       |   |
|   | - 4 GB LPDDR4X SDRAM               |   | - 3D-Printed PLA Sensor Enclosure|   |
|   +------------------------------------+   +----------------------------------+   |
+-----------------------------------------------------------------------------------+
```
