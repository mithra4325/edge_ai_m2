#!/bin/bash
# =====================================================================
# Launcher: M2 Live Enclosure Detection on Raspberry Pi 5
# Runs optimized INT8 ONNX model with live webcam and HUD overlay
# =====================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_PATH="/home/forge/edge-ai-enclosure-detection/venv"

if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "[-] Error: Virtual environment not found at ${VENV_PATH}"
    exit 1
fi

export DISPLAY="${DISPLAY:-:0}"

echo "======================================================================"
echo " Starting M2 Live Detection (INT8 ONNX on Raspberry Pi 5)"
echo " Model:    models/model_v2_int8.onnx (2.72 MB)"
echo " Camera:   /dev/video0 (Logitech C270 HD Webcam)"
echo " Hardware: Raspberry Pi 5 ARM Cortex-A76 CPU (ARM NEON enabled)"
echo " Controls: Press [q] to Quit | [s] to Save Snapshot | [h] to Toggle HUD"
echo "======================================================================"

python3 "${SCRIPT_DIR}/live_inference_m2.py" \
    --model "models/model_v2_int8.onnx" \
    --camera 0 \
    --conf 0.40 \
    --iou 0.45 \
    --save-snapshots
