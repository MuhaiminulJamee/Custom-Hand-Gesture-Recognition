# iOS ONNX Handover Integration Guide

This handover package contains the production-ready gesture model for iOS.

Reference folder:
`handover/v2026.09.10_ios_onnx_01/ios`

## Included files

- `models/gesture_mlp_production.onnx`
- `models/gesture_mlp_onnx_metadata.json`
- `models/gesture_mobile_runtime_config.json`
- `models/gesture_artifact_manifest.json`


## 1) Quick start (what to run)

1. Copy `models/*` into your app’s local model bundle directory.
2. Ensure ONNX Runtime for iOS is installed.
3. Load `gesture_mlp_production.onnx` using ONNX Runtime CPUExecutionProvider.
4. Build frame features (76 floats) and feed as `float32` tensor shape `[1, 76]` named `landmark_features`.
5. Read runtime output and use only `runtime_action` to trigger app behavior.


## 2) Model contract (exact)

From `gesture_mlp_onnx_metadata.json`:

- ONNX format model
- `model_name`: `v18_20 BalancedMLP`
- Input
  - `name`: `landmark_features`
  - `dtype`: `float32`
  - `shape`: `[1, 76]`
- Output class order (8 exposed commands)
  - `left`
  - `right`
  - `up`
  - `down`
  - `open_palm`
  - `like`
  - `dorsal`
  - `ok`
- Open-set/reject signal
  - `known_gesture_mass` (binary-like mass in runtime confidence space)
- Internal class list in metadata also includes
  - `no_gesture` (not exposed as a direct action)

### Feature names and order (76)

1. `landmark_0_x`, `landmark_0_y`
2. `...`
3. `landmark_20_x`, `landmark_20_y`  (21 × 2 = 42 values)
4. `landmark_0_radius` … `landmark_20_radius` (21 values)
5. `index_segment_dx`, `index_segment_dy`, `wrist_index_dx`, `wrist_index_dy`
6. `thumb_index_gap_ratio`
7. `thumb_extension`, `index_extension`, `middle_extension`, `ring_extension`, `pinky_extension`
8. `other_fingers_folded`
9. `thumb_tip_index_mcp_ratio`, `thumb_reach_ratio`

Total = 42 + 21 + 4 + 1 + 6 + 1 + 1 = 76

> Important: feature order **must** match exactly; mismatched ordering produces incorrect predictions.


## 3) Action mapping

Use the mapped action for each runtime gesture. The runtime payload should emit:

- `left` → `Move Left`
- `right` → `Move Right`
- `up` → `Move Up`
- `down` → `Move Down`
- `open_palm` → `Enable / Disable Object Tracking`
- `like` → `Play / Pause`
- `dorsal` → `Return to Default Position`
- `ok` → `Start / Stop Recording`
- `no_gesture` / rejection → `Wait / No Action`

Runtime contract from `MODEL_RELEASE_MANIFEST.json`:
- Consume only `runtime_action` from prediction payload
- Ignore demo command enums in UI-only docs


## 4) Timing and temporal gates (must be kept)

From `gesture_mobile_runtime_config.json`:

- `target_fps`: `10`
- `frame_interval_ms`: `100`
- `action_cooldown_seconds`: `0.8`
- `stable_frames_required`: `3`
- `release_frames_required`: `3`
- `minimum_hold_seconds`: `0.18`

Smoothing/rejection thresholds:

- `ema_alpha`: `0.45`
- `confidence_floor`: `0.80`
- `probability_margin_floor`: `0.18`
- `known_mass_floor`: `0.95` in runtime config, and `known_mass_floor`: `0.80` in metadata validation section

Keep edge filtering/cooldown behavior identical to existing runtime behavior:
- require stable frames before emit
- enforce cooldown per edge action
- gate repeated same-action re-triggers until gesture is released enough frames


## 5) Runtime preprocessing assumptions

To preserve parity with current model behavior, keep the following behavior aligned:

- Use landmarks from hand tracking exactly per frame.
- Keep horizontal semantics consistent with calibrated config:
  - positive index delta maps to **left**
  - negative index delta maps to **right**
- No horizontal mirror in this contract (`mirror_horizontal: false`).
- Do not use .bat scripts in iOS integration.
- Keep low-light / analysis resize settings if your camera pipeline differs from server defaults:
  - recommended analysis resolution: `960x960`
  - if input larger, downscale for analysis while preserving ROI detail


## 6) Recommended implementation flow (mobile side)

1. Capture frame at ~10 FPS.
2. Detect hand landmarks (1 hand only).
3. Convert landmarks to 76D vector in exact order.
4. Create ONNX tensor:
   - shape `[1, 76]`
   - type `float32`
   - name `landmark_features`
5. Run inference.
6. Apply temporal gate:
   - stable frame filtering
   - probability and mass floors
   - hold/cooldown/release edge checks
7. Emit only `runtime_action`.


## 7) Files you should verify before integration

- `ios/models/gesture_mlp_onnx_metadata.json`
- `ios/models/gesture_mobile_runtime_config.json`
- `ios/models/gesture_artifact_manifest.json`
- `ios/MODEL_RELEASE_MANIFEST.json`

Check integrity if needed against checksums in manifest:
- `gesture_mlp_production.onnx` SHA256 in metadata


## 8) Troubleshooting

- If gestures look reversed: check your x/y coordinate orientation and the `horizontal_semantic_swap` calibration assumption.
- If model is noisy: verify feature normalization/order and cooldown/release thresholds.
- If actions are missing: confirm `known_gesture_mass` is flowing through and not bypassed by your own reject logic.
- If no outputs ever appear: verify ONNX Runtime EP loads and model input name exactly equals `landmark_features`.


## 9) Packaging checks

- Keep the 4 model files together in release artifact.
- Avoid embedding external batch scripts (`.bat`) in mobile release flow.
- Ensure app side loads only the iOS-specific artifact directory and does not depend on Python/desktop helpers.


---

This README is the canonical mobile runtime contract for this handover.
Use `runtime_action` only for app actions and preserve current runtime gate timings for stable behavior.