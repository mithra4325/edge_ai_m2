#!/usr/bin/env python3
"""
M2.2 Live Edge AI Enclosure Detection Pipeline (Raspberry Pi 5)
Runs optimized INT8 ONNX model on USB webcam live stream with real-time HUD telemetry.
Model: models/model_v2_int8.onnx (2.72 MB, quantized YOLO26n)
Hardware: Raspberry Pi 5 (ARM Cortex-A76 CPU, ARM NEON acceleration)
"""

import os
import sys
import time
import argparse
from datetime import datetime
import cv2
import numpy as np
import onnxruntime as ort

def get_cpu_temp():
    """Reads Raspberry Pi 5 onboard SoC thermal sensor."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 0.0

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resizes and pads image to square target resolution preserving aspect ratio."""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def postprocess(raw_output, conf_thresh, iou_thresh, ratio, pad):
    """
    Parses YOLO26n ONNX output tensor (1, 5, 8400) -> detections [x1, y1, x2, y2, conf, cls_id].
    """
    dw, dh = pad
    out = raw_output[0].T  # shape (8400, 5): cx, cy, w, h, conf
    scores = out[:, 4]
    mask = scores >= conf_thresh

    if not np.any(mask):
        return []

    boxes = out[mask, :4]
    filtered_scores = scores[mask]

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2 - dw) / ratio
    y1 = (cy - h / 2 - dh) / ratio
    x2 = (cx + w / 2 - dw) / ratio
    y2 = (cy + h / 2 - dh) / ratio

    boxes_xywh = np.stack([(x1).astype(int), (y1).astype(int), (x2 - x1).astype(int), (y2 - y1).astype(int)], axis=1)
    indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), filtered_scores.tolist(), score_threshold=conf_thresh, nms_threshold=iou_thresh)

    detections = []
    if len(indices) > 0:
        for idx in np.array(indices).flatten():
            detections.append({
                "box": [float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])],
                "conf": float(filtered_scores[idx]),
                "class_id": 0,
                "label": "enclosure"
            })
    return detections

def draw_hud(frame, detections, latency_infer_ms, latency_total_ms, fps, temp_c, show_hud=True):
    """Renders high-contrast Edge AI diagnostic Heads-Up Display."""
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Draw detected bounding boxes
    for det in detections:
        box = det["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        conf = det["conf"]
        label = det["label"]

        # Clamp to frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Bounding box (neon cyan-green for high visibility)
        color = (0, 255, 128)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label tag with background pill
        tag_text = f"{label.upper()} {conf * 100:.1f}%"
        (tw, th), baseline = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tag_y1 = max(0, y1 - th - 8)
        tag_y2 = y1
        tag_x2 = min(w - 1, x1 + tw + 10)
        cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
        cv2.putText(annotated, tag_text, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    if not show_hud:
        return annotated

    # HUD Semi-transparent top overlay banner
    banner_h = 100
    sub_img = annotated[0:banner_h, 0:w]
    dark_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 0.35, dark_rect, 0.65, 1.0)
    annotated[0:banner_h, 0:w] = res

    # HUD Text Lines
    # Row 1: Title & Model
    cv2.putText(annotated, "M2 EDGE AI ENCLOSURE DETECTION", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, "Model: YOLO26n INT8 ONNX (2.72 MB)", (w - 380, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    # Row 2: Performance Telemetry
    fps_color = (0, 255, 0) if fps >= 5.0 else (0, 165, 255)
    cv2.putText(annotated, f"FPS: {fps:.1f}", (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2, cv2.LINE_AA)
    cv2.putText(annotated, f"Infer: {latency_infer_ms:.1f} ms", (135, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"Total: {latency_total_ms:.1f} ms", (295, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"CPU Temp: {temp_c:.1f} C", (w - 220, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

    # Row 3: Detection Count & Hardware
    count_color = (0, 255, 128) if len(detections) > 0 else (180, 180, 180)
    cv2.putText(annotated, f"Detected Enclosures: {len(detections)}", (12, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, count_color, 2, cv2.LINE_AA)
    cv2.putText(annotated, "Hardware: Raspberry Pi 5 (Quad Cortex-A76 CPU)", (w - 440, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA)

    # Bottom helper bar
    cv2.putText(annotated, "Keys: [q] Quit  |  [s] Save Frame  |  [h] Toggle HUD", (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return annotated

def main():
    parser = argparse.ArgumentParser(description="M2 Live Enclosure Detection (Raspberry Pi 5)")
    parser.add_argument("--model", type=str, default="models/model_v2_int8.onnx", help="Path to INT8 ONNX model")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (/dev/video0)")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height")
    parser.add_argument("--headless", action="store_true", help="Run without X11 GUI window")
    parser.add_argument("--test-frames", type=int, default=0, help="Automatically exit after N frames (for validation)")
    parser.add_argument("--save-snapshots", action="store_true", help="Auto save snapshots to evidence folder")
    parser.add_argument("--save-dir", type=str, default="M2/02_model_conversion_and_live_deployment/evidence", help="Snapshot output dir")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    model_path = os.path.join(repo_root, args.model)
    save_dir = os.path.join(repo_root, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"[-] Error: Model not found at {model_path}")
        sys.exit(1)

    print("=" * 70)
    print(" M2: LIVE EDGE AI OBJECT DETECTION (RASPBERRY PI 5)")
    print("=" * 70)
    print(f"[*] Target Model:    {model_path}")
    print(f"[*] File Size:       {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
    print(f"[*] Camera Device:   /dev/video{args.camera}")
    print(f"[*] Resolution:      {args.width}x{args.height}")
    print(f"[*] Confidence:      {args.conf * 100:.0f}%")
    print(f"[*] Headless Mode:   {args.headless}")

    print("\n[*] Initializing ONNX Runtime Session (CPUExecutionProvider)...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4  # Utilize all 4 Cortex-A76 cores
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    print(f"[*] Opening webcam device /dev/video{args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[-] Error: Could not open camera on /dev/video{args.camera}")
        sys.exit(1)

    print("[+] Camera initialized successfully.")
    print("[*] Starting live inference loop...")

    show_hud = True
    frame_idx = 0
    t_start = time.time()
    rolling_fps = 0.0
    saved_count = 0

    try:
        while True:
            t_frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[-] Failed to grab frame from camera.")
                break

            frame_idx += 1

            # 1. Preprocess
            t0 = time.time()
            img_proc, ratio, pad = letterbox(frame, (640, 640))
            blob = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[None, ...]
            t_pre_ms = (time.time() - t0) * 1000.0

            # 2. Inference
            t0 = time.time()
            raw_out = session.run(None, {input_name: blob})[0]
            t_infer_ms = (time.time() - t0) * 1000.0

            # 3. Postprocess
            t0 = time.time()
            detections = postprocess(raw_out, args.conf, args.iou, ratio, pad)
            t_post_ms = (time.time() - t0) * 1000.0

            t_total_ms = (time.time() - t_frame_start) * 1000.0
            instant_fps = 1000.0 / t_total_ms if t_total_ms > 0 else 0.0
            rolling_fps = 0.9 * rolling_fps + 0.1 * instant_fps if rolling_fps > 0 else instant_fps
            temp_c = get_cpu_temp()

            annotated_frame = draw_hud(frame, detections, t_infer_ms, t_total_ms, rolling_fps, temp_c, show_hud)

            # Auto-save snapshot for evidence if requested
            if args.save_snapshots and (len(detections) > 0 or frame_idx % 25 == 0) and saved_count < 10:
                snap_path = os.path.join(save_dir, f"m2_live_frame_{frame_idx:04d}.jpg")
                cv2.imwrite(snap_path, annotated_frame)
                saved_count += 1
                print(f"[+] Evidence frame captured: {snap_path} (Detections: {len(detections)}, Infer: {t_infer_ms:.1f}ms)")

            if not args.headless:
                cv2.imshow("M2 Edge AI Enclosure Detection (Pi 5)", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[*] User quit live stream.")
                    break
                elif key == ord('s'):
                    snap_path = os.path.join(save_dir, f"snapshot_manual_{int(time.time())}.jpg")
                    cv2.imwrite(snap_path, annotated_frame)
                    print(f"[+] Saved snapshot to {snap_path}")
                elif key == ord('h'):
                    show_hud = not show_hud
            else:
                if frame_idx % 10 == 0 or len(detections) > 0:
                    print(f"[*] Frame {frame_idx:04d} | Infer: {t_infer_ms:5.1f} ms | FPS: {rolling_fps:4.1f} | Detections: {len(detections)} | Temp: {temp_c:4.1f}°C")

            if args.test_frames > 0 and frame_idx >= args.test_frames:
                print(f"[+] Completed {args.test_frames} validation frames.")
                break

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        elapsed = time.time() - t_start
        print(f"\n[+] Live stream ended: {frame_idx} frames processed in {elapsed:.1f}s ({frame_idx/elapsed:.2f} avg FPS).")

if __name__ == "__main__":
    main()
