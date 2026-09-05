"""Bounded live notebook tests using the exact dashboard inference engine."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import time
import urllib.request
import urllib.error

import numpy as np


def live_camera(root, source='ncm', duration=20, expected_label=None, device=0, display_frames=True):
    """NCM JPEG API, local USB webcam, or Colab browser camera; no auto-training.

    Capture may be 20/30/60 FPS; consume the latest observation at <=10 FPS.
    To measure accuracy, deliberately hold expected_label for the whole test.
    No camera starts merely by importing this function or running notebook QA.
    """
    import cv2
    from backend.config import load_runtime_config
    from backend.model_runtime import InferenceEngine, RuntimeSession
    from IPython.display import display, Image, HTML
    from research.v18_20 import LABELS
    if source not in ['ncm', 'webcam', 'colab']:
        raise ValueError('source must be ncm, webcam, or colab')
    if expected_label is not None and expected_label not in LABELS:
        raise ValueError(f'Expected label must be one of {LABELS}')
    config = load_runtime_config(Path(root) / 'models')
    engine = InferenceEngine(config, Path(root) / 'models')
    session = RuntimeSession(config)
    capture = None
    records = []
    display_handle = display(HTML('Starting camera test...'), display_id=True) if display_frames else None
    try:
        if source == 'webcam':
            capture = cv2.VideoCapture(device)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                raise RuntimeError('Webcam could not open. Check device index and camera permission.')
        elif source == 'ncm':
            request = urllib.request.Request('http://127.0.0.1:8200/api/ncm/connect', method='POST')
            with urllib.request.urlopen(request, timeout=5) as response:
                json.load(response)
        else:
            from google.colab.output import eval_js
            eval_js("""(async () => {
              if (window.gestureTestStream) window.gestureTestStream.getTracks().forEach(t => t.stop());
              const host = document.createElement('div');
              const video = document.createElement('video'); video.autoplay = true; video.width = 480;
              const stop = document.createElement('button'); stop.textContent = 'Stop camera test';
              window.gestureTestStopped = false;
              const stream = await navigator.mediaDevices.getUserMedia({video:true,audio:false});
              window.gestureTestStream = stream; video.srcObject = stream;
              stop.onclick = () => { window.gestureTestStopped = true; stream.getTracks().forEach(t => t.stop()); };
              host.append(video, stop); document.body.append(host); await video.play();
              window.gestureTestVideo = video; window.gestureTestHost = host;
              return true;
            })()""")
        started = time.perf_counter()
        previous_start = None
        last_frame_id = None
        while time.perf_counter() - started < duration:
            if previous_start is not None:
                time.sleep(max(0, .1 - (time.perf_counter() - previous_start)))
            previous_start = time.perf_counter()
            if source == 'ncm':
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8200/api/ncm/frame.jpg', timeout=3) as response:
                        frame_id = response.headers.get('X-NCM-Frame-ID')
                        payload = response.read()
                    if frame_id == last_frame_id:
                        continue
                    last_frame_id = frame_id
                except Exception as error:
                    if (isinstance(error, urllib.error.HTTPError) and error.code == 503
                            and time.perf_counter() - started < 5):
                        continue  # Allow the board connection to deliver its first JPEG.
                    raise RuntimeError('NCM camera unavailable; connect the board and run the local backend.') from error
            elif source == 'webcam':
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError('Webcam stopped returning frames.')
                ok, encoded = cv2.imencode('.jpg', frame)
                if not ok:
                    continue
                payload = encoded.tobytes()
            else:
                data_url = eval_js("""(() => {
                  if (window.gestureTestStopped) return null;
                  const v=window.gestureTestVideo, c=document.createElement('canvas');
                  c.width=v.videoWidth; c.height=v.videoHeight; c.getContext('2d').drawImage(v,0,0);
                  return c.toDataURL('image/jpeg',0.85);
                })()""")
                if data_url is None:
                    break
                payload = base64.b64decode(data_url.split(',')[1])
            result = engine.process_frame(payload, session, center_crop_ratio=config.roi_size_ratio)
            timing = result.get('timing', {})
            record = dict(elapsed_s=time.perf_counter() - started, expected=expected_label,
                          prediction=result.get('runtime_prediction', 'no_gesture'), status=result.get('status'),
                          confidence=result.get('confidence', 0), known_mass=result.get('known_gesture_mass', 0),
                          action=result.get('runtime_action'), reason=result.get('action_reason', result.get('message', '')),
                          total_ms=timing.get('total_ms', 0), mediapipe_ms=timing.get('mediapipe_ms', 0),
                          classifier_ms=timing.get('classifier_ms', 0))
            records.append(record)
            if display_handle is not None:
                frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                height, width = frame.shape[:2]; side = round(min(height, width) * config.roi_size_ratio)
                left, top = (width - side) // 2, (height - side) // 2
                for x, y in result.get('landmarks', []):
                    cv2.circle(frame, (int(left + x * side), int(top + y * side)), 3, (255, 220, 30), -1)
                cv2.putText(frame, f"{record['prediction']} {record['confidence']:.0%} | {record['total_ms']:.1f} ms", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (40, 255, 220), 2)
                display_handle.update(Image(data=cv2.imencode('.jpg', frame)[1].tobytes()))
    finally:
        if capture is not None:
            capture.release()
        engine.close()
        if source == 'colab':
            from google.colab.output import eval_js
            eval_js("""(() => { if(window.gestureTestStream) window.gestureTestStream.getTracks().forEach(t=>t.stop());
              if(window.gestureTestHost) window.gestureTestHost.remove(); return true; })()""")
    import pandas as pd
    return pd.DataFrame(records)
