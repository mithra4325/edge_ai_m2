# M2.1: Model Export to ONNX (3 Marks)

**Assessment Requirement:**  
> *"I run the ONNX export myself and show my commit"*

---

## 1. Objective and Overview

In Milestone 2, we upgrade our Edge AI deployment pipeline by converting our trained PyTorch enclosure detection model (`object-detection-n6b60-2-yolo26n-t2`, saved as `models/model_v2_original.pt`) into the open, interoperable **Open Neural Network Exchange (ONNX)** format (`models/model_v2.onnx`).

Exporting to ONNX is a necessary foundational step for Edge AI deployment on embedded platforms like the Raspberry Pi 5 because:
1. **Framework Decoupling:** It decouples inference from heavy training frameworks (PyTorch, Python autograd engines).
2. **Graph Optimization:** It enables operator fusion, dead-code elimination, and constant folding via tools like `onnxslim`.
3. **Hardware Runtime Target:** It allows execution directly inside `onnxruntime` optimized with ARM NEON vector instructions for the Raspberry Pi 5's Cortex-A76 processor.
4. **Quantization Pathway:** It serves as the standard intermediate representation (IR) required for dynamic and static INT8 quantization.

---

## 2. Model Specifications

| Parameter | PyTorch Source (`model_v2_original.pt`) | Exported ONNX (`model_v2.onnx`) |
| :--- | :--- | :--- |
| **Model Origin** | Roboflow: `object-detection-n6b60-2-yolo26n-t2` | Cleanly exported from PyTorch |
| **Architecture** | YOLO Nano (yolo26n decoupled head) | Slimmed ONNX Computational Graph |
| **Parameters** | 2,375,031 (fused) | 2,375,031 |
| **Computational Cost** | 5.3 GFLOPs | 5.3 GFLOPs |
| **Precision** | Float32 (FP32) | Float32 (FP32) |
| **File Size** | **5.17 MB** | **9.34 MB** (uncompressed graph + weights) |
| **Input Node** | Tensor `(1, 3, 640, 640)` (BCHW, RGB normalized) | `images` `[1, 3, 640, 640]` float32 |
| **Output Node** | Decoupled bounding boxes + confidence | `output0` `[1, 5, 8400]` float32 |
| **Target Classes** | 1 class (`enclosure`) | 1 class (`enclosure`) |
| **Opset Version** | N/A | **ONNX Opset 17** |

---

## 3. How to Run the ONNX Export

Activate the virtual environment and execute the dedicated export script:

```bash
# 1. Activate virtual environment
source /home/forge/edge-ai-enclosure-detection/venv/bin/activate

# 2. Navigate to M2 root
cd /home/forge/edge_ai_m2

# 3. Run the export script
python3 M2/01_onnx_export/export_onnx.py
```

### Script Execution Summary:
- Loads `models/model_v2_original.pt`.
- Invokes Ultralytics PyTorch-to-ONNX exporter with `opset=17`, `imgsz=640`, and `simplify=True` (`onnxslim`).
- Validates the exported ONNX model graph structure using `onnx.checker.check_model`.
- Inspects input and output tensor shapes.
- Performs an inference test on a real physical test image (`data/test/images/IMG_9716_HEIC...jpg`) to prove numerical validity.

---

## 4. Verification and Parity Proof

### Graph Integrity Verification:
The exported model was validated against the ONNX standard:
```python
import onnx
model = onnx.load("models/model_v2.onnx")
onnx.checker.check_model(model)
print("Graph topology valid: PASS")
```
- **Graph Status:** Well-formed DAG (Directed Acyclic Graph) with valid topological ordering.
- **Input Node Name:** `images` (Shape: `[1, 3, 640, 640]`, Type: `FLOAT`)
- **Output Node Name:** `output0` (Shape: `[1, 5, 8400]`, Type: `FLOAT` representing 8400 anchor predictions: 4 bbox coordinates `[cx, cy, w, h]` + 1 class score `[enclosure]`).

### Detection Parity on Real Physical Test Frame:
- **Test Image:** `IMG_9716_HEIC.rf.3c8813683d80285ad53efe2e0d31c566.jpg` (3D-printed enclosure on desk).
- **PyTorch Original Detection:**
  - Class: `enclosure` (id: 0)
  - Confidence: **96.77%** (`0.9677`)
  - Bounding Box: `[983.8, 832.7, 2859.1, 1992.0]`
- **ONNX Runtime Output:**
  - Output Tensor shape: `(1, 5, 8400)`
  - Maximum raw class confidence across anchors: **96.77%**
  - **Delta:** 0.00% difference in predicted confidence.

The complete automated run log is preserved in [`export_log.txt`](export_log.txt).

---

## 5. Student Checklist for Milestone Proof

To demonstrate this task for college assessment:
1. Show the `models/model_v2.onnx` file generated on the device.
2. Open terminal and run `python3 M2/01_onnx_export/export_onnx.py` to show live generation.
3. Show the Git commit in the GitHub repository history containing `M2/01_onnx_export/` and `models/model_v2.onnx`.
4. Point to the terminal output showing `opset 17` and output shape `(1, 5, 8400)`.
