# Edge AI Deployment Guide (Raspberry Pi 5)

This guide provides end-to-end instructions for deploying and running the optimized INT8 Edge AI Enclosure Detector on a physical Raspberry Pi 5.

---

## 1. System Requirements

### Hardware:
- **Single Board Computer:** Raspberry Pi 5 (4 GB or 8 GB RAM)
- **Camera Sensor:** USB UVC Webcam (e.g., Logitech C270 HD Webcam on `/dev/video0`)
- **Power Supply:** Official Raspberry Pi 27W USB-C Power Supply (5.1V / 5.0A)
- **Cooling:** Raspberry Pi Active Cooler or heatsink fan
- **Storage:** MicroSD Card or NVMe SSD (minimum 16 GB with >= 4 GB free)

### Software & Operating System:
- **OS:** Debian GNU/Linux 13 (Trixie) 64-bit / Raspberry Pi OS 64-bit (`aarch64`)
- **Python:** Python 3.11, 3.12, or 3.13
- **Video Subsystem:** Video4Linux2 (`v4l2`)

---

## 2. Environment Setup

Clone the repository and set up a Python virtual environment:

```bash
# 1. Clone M2 repository
git clone https://github.com/mithra4325/edge_ai_m2.git
cd edge_ai_m2

# 2. Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip and install required dependencies
pip install --upgrade pip
pip install opencv-python numpy onnx onnxruntime psutil
```

---

## 3. Verify Hardware & Camera

Check that the USB webcam is recognized by the Linux kernel:

```bash
# List video devices
v4l2-ctl --list-devices

# Verify USB bus enumeration
lsusb | grep -i "webcam\|logitech"
```
The webcam will typically enumerate as `/dev/video0` and `/dev/video1`.

---

## 4. Run Live Detection

Launch the live detection pipeline with Heads-Up Display:

```bash
# Option A: Using the one-click launcher shell script
bash M2/02_model_conversion_and_live_deployment/run_m2_live.sh

# Option B: Direct Python execution
python3 M2/02_model_conversion_and_live_deployment/live_inference_m2.py \
    --model "models/model_v2_int8.onnx" \
    --camera 0 \
    --conf 0.40 \
    --iou 0.45
```

### Interactive Keyboard Controls:
- `q`: Quit the live stream safely.
- `s`: Save a timestamped snapshot of the live frame into `M2/02_model_conversion_and_live_deployment/evidence/`.
- `h`: Toggle HUD overlay elements on and off.

---

## 5. Running the Validation Benchmarks

To reproduce all benchmarks on your device:

```bash
# 1. Run accuracy comparison across all models
python3 M2/03_accuracy_comparison/evaluate_accuracy_comparison.py

# 2. Run fair 20 warmup + 100 timed runs latency benchmark
python3 M2/04_fair_benchmark/run_fair_benchmark.py
```
