# Model Card — Gesture MLP ONNX Runtime Release

## Purpose

Local static-hand gesture recognition with a dedicated temporal Palm → Fist →
Palm Follow Object procedure and a strict 10 FPS application-capacity test.

## Runtime architecture

`camera/image → MediaPipe Hand Landmarker → 21 landmarks → 76 float32 features →
ONNX MLP → 15 probabilities → geometry resolver → EMA/stability gate → mapped action`

The Windows demonstration uses browser webcam capture and passes mirror-corrected
frames into this unchanged model pipeline.

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
- Performance and accuracy should be checked on the target camera, lighting,
  distance, skin tones, backgrounds, and hardware.
- The application emits semantic mapped-action labels; OS automation is outside
  this package.
- ONNX is immutable at runtime. Feedback requires an offline retraining and
  qualified re-export cycle.
- Localhost deployment is for a trusted PC. Do not expose ports directly to an
  untrusted network.
