# Model Card — Gesture MLP ONNX + NCM Camera Release

## Purpose

Local static-hand gesture recognition with a dedicated temporal Palm → Fist →
Palm Follow Object procedure and a strict 10 FPS application-capacity test.

## Runtime architecture

`board JPEG → USB-NCM → JLIP/TCP parser → MediaPipe Hand Landmarker → 21 landmarks → 76 float32 features → ONNX MLP → 15 probabilities → geometry resolver → EMA/stability gate → mapped action`

This edition does not use browser webcam capture. The backend binds its outgoing
camera socket to `192.168.50.1`, connects to `192.168.50.2:5000`, validates JLIP
lengths and CRC32, and supplies board JPEGs to the unchanged model pipeline.

Follow Object uses the same ONNX classifier plus a state machine. It is not a
separate dynamic neural network.

## Model contract

- File: `gesture_mlp_production.onnx`
- Execution provider: ONNX Runtime `CPUExecutionProvider`
- Input: `landmark_features`, float32, shape `[N, 76]`
- Output: float probabilities, shape `[N, 15]`
- Classes: call, rock, like, ok, one, one_down, one_left, one_right, palm,
  peace, dorsal_hand, fist, zoom_in, zoom_out, no_gesture
- ONNX opsets: `ai.onnx` 16 and `ai.onnx.ml` 1
- Source family: selected offline MLP

## Conversion evidence

- Preserved validation samples: 256
- Prediction agreement: 1.0
- Maximum absolute probability error: approximately `1.3113e-6`
- Required minimum agreement: 0.99
- Allowed maximum probability error: `1e-4`

The runtime verifies the ONNX SHA-256, feature order, class order, input type and
shape, probability output shape, finite values, normalization, and parity flag
before becoming ready.

## Limitations

- `no_gesture` quality improvement is explicitly deferred.
- Accuracy must be checked on the board camera, lighting, distance, skin tones,
  backgrounds, lens, exposure, and orientation.
- Live operation depends on working board ARP, USB-NCM NTB parsing, UDP discovery,
  TCP accept, and valid JLIP/JPEG transmission; those firmware components are not
  part of the ONNX model.
- The application emits semantic mapped-action labels and implements its local
  uploaded-media action demo. Product/app integration remains the app developer's
  responsibility.
- ONNX is immutable at runtime. Feedback requires an offline retraining and
  qualified re-export cycle.
- Localhost deployment is for a trusted PC. Do not expose ports directly to an
  untrusted network.
