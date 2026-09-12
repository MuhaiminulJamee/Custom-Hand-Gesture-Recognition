# Stage 2: Gesture ONNX Model Package (iOS)

This package contains the trained 8-class MLP ONNX model, metadata, and runtime configuration.

## Contents
- `models/gesture_mlp_production.onnx`: ONNX float32 classifier (Opset 16).
- `models/gesture_mlp_onnx_metadata.json`: Input/output shapes, tensor names, opset, and checksum.
- `models/gesture_mobile_runtime_config.json`: Production runtime thresholds and timing parameters.
- `models/gesture_artifact_manifest.json`: Checksum manifest for integrity verification.
- `contracts/MODEL_RELEASE_MANIFEST.json`: Release contract specification.

## ONNX Model Contract

### Inputs
- **Name:** `landmark_features`
- **Type:** `float32`
- **Shape:** `[1, 76]` (produced by `GestureFeatureExtractor.landmarksToFeature()`)

### Outputs (Query By Name)
The ONNX model has **3 output tensors**. In Swift ONNX Runtime, query `probabilities` and `known_gesture_mass` by name:
1. `probabilities`: Shape `[1, 8]`, `float32`. Contains softmax probabilities for the 8 classes.
2. `known_gesture_mass`: Shape `[1, 1]`, `float32`. Confidence score indicating whether the input matches any known gesture.
3. `label`: Internal integer/string prediction label.

```swift
// Swift ONNX Runtime example:
let outputs = try session.run(
    withInputs: ["landmark_features": inputTensor],
    outputNames: ["probabilities", "known_gesture_mass"],
    runOptions: nil
)
```

### Canonical 8-Class Output Order (Indices 0..7)
```text
Index 0: left        -> Move Left
Index 1: right       -> Move Right
Index 2: up          -> Move Up
Index 3: down        -> Move Down
Index 4: open_palm   -> Enable / Disable Object Tracking
Index 5: like        -> Play / Pause
Index 6: dorsal      -> Return to Default Position
Index 7: ok          -> Start / Stop Recording
```

### Rejection Sentinel (`no_gesture` -> "Wait / No Action")
`no_gesture` is **not** an ONNX output index. It is returned when:
- No hand is detected by MediaPipe.
- `known_gesture_mass < 0.95` (open-set outlier rejection).
- Top probability $< 0.80$ (confidence floor).

### Timing & State Machine
- **Target Frame Rate:** ~10 FPS (100 ms interval).
- **Hold Time:** Require the gesture to be held stable for **3 consecutive frames** before triggering an action.
- **Cooldown:** Enforce a **0.80s cooldown** after firing an action before allowing repeat actions.
