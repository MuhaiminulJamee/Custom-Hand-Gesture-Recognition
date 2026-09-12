#!/usr/bin/env python3
"""
Standalone Real-Time Gesture Recognition with ONNX
Runs directly in an OpenCV window on your PC without starting the web server.

Usage:
  python scripts/run_live_camera.py                 # Uses default PC webcam
  python scripts/run_live_camera.py --source ncm    # Connects to USB-NCM Board Camera (192.168.50.2:5000)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure thread limit for OpenBLAS on Windows
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from backend.config import load_runtime_config
from backend.model_runtime import InferenceEngine, RuntimeSession
from backend.ncm_camera import NcmCameraClient, NcmCameraConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Live ONNX Gesture Recognition")
    parser.add_argument(
        "--source",
        choices=["webcam", "ncm"],
        default="webcam",
        help="Video source: 'webcam' (default) or 'ncm' (development board)",
    )
    parser.add_argument(
        "--webcam-id",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    args = parser.parse_args()

    print("Loading models and runtime configuration...")
    config = load_runtime_config()
    engine = InferenceEngine(config)
    session = RuntimeSession(config)

    status = engine.status()
    if not status.get("ready"):
        print(f"Error: InferenceEngine is not ready: {status}")
        return

    print("Models loaded successfully!")
    print("Press 'q' or 'ESC' in the camera window to exit.\n")

    if args.source == "ncm":
        print("Connecting to USB-NCM camera at 192.168.50.2:5000...")
        ncm_config = NcmCameraConfig(host_ip="192.168.50.1", device_ip="192.168.50.2", tcp_port=5000)
        client = NcmCameraClient(ncm_config)
        client.start()
        
        window_title = "ONNX Gesture Recognition - USB-NCM Camera (Press 'q' to exit)"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 960, 720)

        try:
            while True:
                frame_data = client.get_frame(timeout=0.1)
                if frame_data is None or not frame_data.jpeg:
                    # Check connection status
                    if not client.connected:
                        display = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(
                            display,
                            "Connecting to NCM Camera (192.168.50.2:5000)...",
                            (30, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 165, 255),
                            2,
                        )
                        cv2.imshow(window_title, display)
                        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                            break
                        continue
                    continue

                jpeg_bytes = frame_data.jpeg
                result = engine.process_frame(jpeg_bytes, session)
                
                # Decode for display
                frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    _draw_overlay(frame, result)
                    cv2.imshow(window_title, frame)

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            client.stop()
            cv2.destroyAllWindows()
            engine.close()

    else:
        print(f"Opening PC Webcam (Index {args.webcam_id})...")
        cap = cv2.VideoCapture(args.webcam_id)
        if not cap.isOpened():
            print(f"Error: Could not open webcam at index {args.webcam_id}.")
            return

        window_title = "ONNX Gesture Recognition - PC Webcam (Press 'q' to exit)"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 960, 720)

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Failed to grab frame from webcam.")
                    break

                # Encode frame to JPEG bytes for InferenceEngine
                success, encoded = cv2.imencode(".jpg", frame)
                if not success:
                    continue

                result = engine.process_frame(encoded.tobytes(), session)
                _draw_overlay(frame, result)
                cv2.imshow(window_title, frame)

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            engine.close()


def _draw_overlay(frame: np.ndarray, result: dict) -> None:
    """Draws bounding boxes, gesture label, and action status on the OpenCV frame."""
    h, w = frame.shape[:2]
    status = result.get("status", "")
    pred = result.get("runtime_prediction", "no_gesture")
    action = result.get("runtime_action", "Wait / No Action")
    reason = result.get("action_reason", "")
    conf = result.get("confidence", 0.0)
    mass = result.get("known_gesture_mass", 0.0)
    fps = result.get("actual_fps", 0.0)

    # Top banner background
    cv2.rectangle(frame, (0, 0), (w, 85), (20, 20, 20), -1)

    # Text color based on detection
    if pred != "no_gesture" and conf >= 0.8:
        pred_color = (50, 220, 50)  # Green
    else:
        pred_color = (180, 180, 180)  # Gray

    # 1. Gesture prediction & confidence
    pred_text = f"Gesture: {pred.upper()} ({conf*100:.1f}%)"
    cv2.putText(frame, pred_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, pred_color, 2)

    # 2. Action status
    if action != "Wait / No Action":
        action_color = (0, 230, 230)  # Yellow
    else:
        action_color = (160, 160, 160)
    action_text = f"Action: {action}"
    if reason:
        action_text += f" ({reason})"
    cv2.putText(frame, action_text, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, action_color, 2)

    # 3. Stats on top right
    stats_text = f"Mass: {mass:.2f} | FPS: {fps:.1f}"
    cv2.putText(frame, stats_text, (w - 240, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


if __name__ == "__main__":
    main()

