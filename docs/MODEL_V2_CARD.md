# Model Card: YOLO26n Enclosure Detector V2

## Model Details
- **Model Name:** `object-detection-n6b60-2-yolo26n-t2`
- **Model Version:** 2.0 (Milestone 2)
- **Model Type:** Object Detection
- **Architecture:** YOLO Nano (yolo26n decoupled head with Anchor-Free Anchor Assignment)
- **Trained By:** Mithra Ravi (`mithra4325`)
- **Training Platform:** Roboflow Train (`mithu-1vntk/object-detection-n6b60`)
- **Base Weights:** YOLO26n pretrained weights
- **Number of Parameters:** 2,375,031 parameters (fused)
- **Computational Cost:** 5.3 GFLOPs @ 640x640 resolution
- **Target Hardware:** Raspberry Pi 5 (ARM Cortex-A76 CPU @ 2.4 GHz)

---

## Intended Use
- **Primary Use Case:** Automated visual quality control, presence verification, and spatial counting of custom 3D-printed PIR sensor enclosures on manufacturing workbenches and 3D-printer beds.
- **Intended Users:** Edge AI engineers, robotics integrators, quality assurance automation systems.
- **Out of Scope Uses:** Generic multi-class object detection; outdoor nighttime surveillance.

---

## Training Data & Augmentations
- **Dataset Version:** Dataset v2 (`object-detection-n6b60/2`)
- **Total Images:** 101 annotated physical images (173 enclosure bounding boxes)
- **Split Ratio:**
  - **Train:** 71 images (70.3%)
  - **Validation:** 20 images (19.8%)
  - **Test:** 10 images (9.9%)
- **Data Augmentations Applied:**
  - Auto-orient with EXIF stripping
  - Horizontal flip (50% probability)
  - Random brightness adjustment (-15% to +15%)
  - Random blur (up to 1.5 pixels)
  - Bounding box jitter and scale variations

---

## Training Hyperparameters & Evolution
- **Epochs Trained:** 305 epochs
- **Input Image Size:** 640x640 pixels
- **Batch Size:** Dynamic edge batching
- **Optimizer:** AdamW with cosine learning rate schedule
- **Loss Functions:** Complete IoU (CIoU) bounding box loss + Distribution Focal Loss (DFL) + Binary Cross-Entropy (BCE) classification loss

---

## Exported Formats & Checkpoints
1. `models/model_v2_original.pt`: PyTorch FP32 checkpoint (5.17 MB, MD5: `ed26737f31d3f12d63cf3d27e9a4515c`)
2. `models/model_v2.onnx`: Exported ONNX Opset 17 FP32 computational graph (9.34 MB)
3. `models/model_v2_int8.onnx`: Quantized Dynamic INT8 ONNX graph (2.72 MB, 70.9% reduction)

---

## Performance Metrics on Identical Test Set (10 images, 16 instances)
- **Precision:** 100.00% (Zero false positives)
- **Recall:** 93.75%
- **F1-Score:** 0.9677
- **mAP@0.50:** 95.80%
- **mAP@0.5:0.95:** 81.91%
- **Edge Latency (Pi 5 CPU):** ~160 - 185 ms per frame (~5.5 FPS)
- **Runtime RAM:** < 60 MB
