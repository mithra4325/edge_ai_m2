# M2.6: Targets, Results, and Engineering Trade-offs (3 Marks)

**Assessment Requirement:**  
> *"I show the accuracy and speed target I wrote before training, my result, and explain the trade-off in simple words"*

---

## 1. Pre-Training Targets (Formulated Before M2 Training)

Before training Dataset v2 on Roboflow and converting the model for edge deployment, we formally established five explicit engineering targets. These targets were designed to address the empirical limitations discovered during Milestone 1:

1. **Target 1 — False Positive Elimination (Precision Target):**
   - *M1 Issue:* M1 exhibited spurious false-positive detections on plain white tabletops and reflective plastic backgrounds (Precision: 93.9%).
   - *M2 Target:* Increase dataset diversity with 100+ images across varied lighting and angles; achieve **Precision $\ge$ 98.0%**.
2. **Target 2 — Detection Accuracy (mAP Target):**
   - *M2 Target:* Maintain **mAP@0.50 $\ge$ 90.0%** across varied distances and angles.
3. **Target 3 — Edge Footprint & Storage (Model Size Target):**
   - *M2 Target:* Compress the deployed model file size to **$\le$ 3.0 MB** (a >50% reduction from the 9.34 MB uncompressed ONNX export) to minimize flash storage and memory bus contention.
4. **Target 4 — Inference Latency & Framerate Target:**
   - *M2 Target:* Achieve single-frame CPU inference latency of **$\le$ 200 ms** (corresponding to **$\ge$ 5.0 FPS**) on the Raspberry Pi 5 without external accelerator hardware.
5. **Target 5 — Runtime Memory & Thermal Stability:**
   - *M2 Target:* Eliminate heavy PyTorch autograd graph overhead; keep total runtime RAM footprint under **100 MB** and maintain SoC temperature under **65°C**.

---

## 2. Targets vs. Actual Results

The table below contrasts our pre-optimization targets against the empirical measurements recorded on the physical Raspberry Pi 5:

| Metric | Pre-Training Target | M1 Baseline (`model_v1`) | M2 FP32 (`model_v2`) | M2 INT8 ONNX *(Achieved)* | Target Met? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Precision** | $\ge 98.0\%$ | 93.9% (1 FP) | 99.7% (0 FP) | **100.0% (0 False Positives)** | **YES (Exceeded)** |
| **mAP@0.50** | $\ge 90.0\%$ | 99.1% | 96.0% | **95.8%** | **YES (Exceeded)** |
| **Model Size** | $\le 3.0\text{ MB}$ | 5.17 MB (PT) | 9.34 MB (ONNX) | **2.72 MB (INT8 ONNX)** | **YES (70.9% cut)** |
| **Inference Latency** | $\le 200\text{ ms}$ | ~220 ms (PT) | ~190 ms (ONNX) | **~160 – 185 ms (NEON INT8)** | **YES** |
| **Framerate (FPS)** | $\ge 5.0\text{ FPS}$ | ~4.5 FPS | ~5.1 FPS | **~5.4 – 6.0 FPS** | **YES** |
| **Runtime RAM** | $< 100\text{ MB}$ | ~520 MB (PyTorch) | ~140 MB | **< 60 MB (ONNX Runtime)** | **YES (88% less)** |
| **CPU Temperature** | $< 65.0^\circ\text{C}$ | 58.2°C | 54.5°C | **52.4°C – 54.0°C** | **YES (Cool & stable)** |

---

## 3. Engineering Trade-offs Explained in Simple Words

### Trade-off 1: Precision vs. Model Size (FP32 to INT8 Quantization)
- **The Concept:** Standard neural network weights use 32-bit floating point numbers (FP32), which can represent tiny mathematical fractions. Quantization converts these numbers into compact 8-bit integers (INT8) ranging from -128 to +127.
- **The Trade-off:** By discarding 24 bits per weight, you risk rounding errors that might slightly reduce detection confidence.
- **Why it worked for us:** Our custom 3D-printed enclosure has distinct visual geometry (chamfered borders, PIR sensor dome cutout, mounting tabs). Dynamic quantization reduced model size from **9.34 MB to 2.72 MB** (saving 70.9% flash storage and cutting memory bandwidth 4x) while detection confidence dropped by less than 0.1% (96.83% vs 96.77%). The accuracy loss was virtually undetectable, making the trade-off overwhelmingly positive.

### Trade-off 2: Speed vs. Image Resolution (640x640 vs. Downscaling)
- **The Concept:** Downscaling input frames from 640x640 to 320x320 would reduce FLOPs by 4x, increasing framerate to ~15 FPS.
- **The Trade-off:** However, 3D-printed enclosures have subtle layer lines and small screw holes. At 320x320, distant enclosures become blurry blobs, increasing false negatives.
- **Our Decision:** We preserved the full **640x640 resolution**. At ~5.5 FPS, the system processes a new inspection frame every ~180 ms. For quality control on a 3D-printer bed or assembly workbench, 5.5 FPS is plenty fast, whereas missed detections would ruin automated inspection.

### Trade-off 3: CPU Optimization vs. Adding Hardware Accelerators
- **The Concept:** We could have attached a Google Coral Edge TPU ($60) or Hailo-8 AI HAT ($70) to achieve 30+ FPS.
- **The Trade-off:** External accelerators add financial cost, mechanical bulk, increased power consumption, and proprietary compiler toolchains that often reject certain neural network operations.
- **Our Decision:** By running ARM NEON-optimized INT8 ONNX Runtime directly on the onboard Broadcom BCM2712 CPU, we achieved our 5+ FPS target with **$0 in extra hardware**, zero extra power consumption, and maximum software portability.
