# Model Card — Eight-Gesture MLP ONNX + NCM Camera Release

Current release: **v18_20**. See [V18_20_README.md](V18_20_README.md) for current training data, notebook, metrics, and limitations. Prior-release statistics below are historical.

## Purpose

Local static-hand command recognition from a USB-NCM development-board camera.
The release exposes exactly eight command probabilities, uses an open-set
rejection sentinel for unsafe frames, and enforces a backend-wide ceiling of 10
inference starts per second with a 100 ms frame budget.

This model is intended for local interactive control after installation-specific
live validation. It is not a safety-critical controller, identity/biometric
system, or a substitute for physical controls and fail-safe behavior.

## Runtime architecture

`board JPEG → USB-NCM → JLIP/TCP validation → adaptive image-quality preprocessing → MediaPipe VIDEO Hand Landmarker → filtered 21-point landmarks → 76 float32 features → eight-output ONNX MLP + known-mass score → pose/open-set/EMA gates → mapped action`

The input image and landmarks are not horizontally flipped. NCM horizontal
calibration swaps the legacy source-coordinate interpretation: decreasing x is
reported as Right and increasing x is reported as Left. This matches the
front-facing operator's intended direction while preserving the original camera
pixels. Once resolved, semantic `left` and `right` labels map normally to Move
Left and Move Right.

## Model contract

- File: `models/gesture_mlp_production.onnx`
- Execution provider: ONNX Runtime `CPUExecutionProvider`
- Input: `landmark_features`, float32, shape `[N, 76]`
- Probability output: float, shape `[N, 8]`
- Auxiliary output: known-gesture mass, shape `[N, 1]`
- Output order: `left`, `right`, `up`, `down`, `open_palm`, `like`, `dorsal`, `ok`
- Rejection/feedback sentinel: `no_gesture` (not an ONNX output class)
- ONNX opset: `ai.onnx` 16
- Source family: balanced (128, 64) MLP with eight exposed commands and an internal unknown class

| Runtime label | Action |
|---|---|
| `left` | Move Left |
| `right` | Move Right |
| `up` | Move Up |
| `down` | Move Down |
| `open_palm` | Enable / Disable Object Tracking |
| `like` | Play / Pause |
| `dorsal` | Return to Default Position |
| `ok` | Start / Stop Recording |

## Rejection and temporal policy

`no_gesture` is returned when no usable hand is detected or a candidate fails
known-mass, confidence, probability-margin, geometry, or temporal checks. It
does not map to an action. The configured runtime uses:

- probability EMA alpha `0.45`;
- confidence floor `0.80`;
- top-class margin floor `0.18`;
- known-gesture mass floor `0.95`;
- three stable frames before execution;
- three release frames and a `0.80 s` action cooldown.

Directional and pose-specific hard geometry vetoes reduce confusion between
similar shapes. Dorsal versus Down is checked using visible 2-D finger direction
and extension. The feature contract contains no depth channel, so this separation
must be challenged during live acceptance rather than assumed from offline data.

## Camera quality and landmark stability

- MediaPipe runs in VIDEO mode with strictly increasing monotonic timestamps.
- Underexposed frames receive graded luminance-based CLAHE/gamma enhancement;
  normally exposed frames are left unchanged.
- Near-blank frames are rejected before MediaPipe and reset the temporal gate.
- Invalid, tiny, clipped, or implausible landmark sets cannot reach the
  classifier.
- Accepted landmarks use per-session velocity-adaptive EMA smoothing. Isolated
  large jumps are rejected; a persistent jump is reacquired to permit genuine
  hand motion.
- Processing preserves source orientation throughout; neither the NCM preview
  nor feature path performs horizontal mirroring.

These filters mitigate dark frames, jitter, and false activation. They do not
guarantee performance for every lens, exposure, motion blur, background, hand
shape, or mounting geometry.

## Safe reviewed online learning

The base ONNX graph is immutable. Safe Learn applies a bounded local NumPy
residual adapter after base inference. It accepts exactly one explicitly reviewed
76-D sample at a time, tries bounded influence levels, and rejects the candidate
if validation/replay metrics or unknown-class behavior regress beyond configured
limits. Accepted adapter state is written atomically and backed up for rollback.
Unsafe/forced learning is unsupported.

A reviewed `no_gesture` sample creates a local rejection prototype that reduces
known-gesture acceptance near that feature vector; it does not add a ninth
classifier output. Save-only feedback remains available for offline dataset
review. A materially new base model still requires offline training, evaluation,
ONNX export, parity checks, and a new signed artifact manifest.

## Offline qualification evidence

The current metadata records:

- 1,769 known-class test samples;
- known-class accuracy `0.99152`;
- macro F1 `0.98928`;
- minimum per-class F1 `0.97473`;
- known acceptance rate `0.97230` at known-mass floor `0.80`;
- 1,146 held-out unknown samples;
- unknown false-acceptance rate `0.01832`;
- accepted-known accuracy `0.99593`;
- ONNX/source prediction agreement `1.0` on 256 parity samples;
- maximum absolute ONNX probability error about `1.05e-7`.

These measurements validate the preserved offline arrays and ONNX conversion.
They are not measurements from the physical NCM camera and must not be presented
as live-device accuracy.

## Required live NCM validation

Before acceptance, collect per-class attempts for all eight commands using the
actual board, lens, firmware, placement, and 10 FPS inference path. The camera
transport may deliver faster than 10 FPS, but inference must remain capped at
10 FPS with its 100 ms frame budget. Cover
normal and low light, varied backgrounds and distances, both hands where
supported, skin tones, hand sizes, and finger thicknesses. Include explicit
confusion challenges for Left/Right, Dorsal/Down, Down/non-command poses, and
OK/Open Palm, plus empty scenes and partial hands.

Record the confusion matrix, per-class precision/recall/F1, false activations per
minute with no command shown, landmark jitter, action latency, observed camera
FPS, frame-budget pass rate, and JLIP/CRC/sequence errors. Offline qualification
is complete only as a software gate; live NCM acceptance remains a separate
hardware test.

## Limitations and operational controls

- Live use depends on correct board ARP, USB-NCM NTB parsing, UDP discovery, TCP
  service, JLIP framing, and JPEG delivery; these are outside the ONNX model.
- Low light can be enhanced, but severe blur, saturation, occlusion, or a nearly
  black frame may still require rejection rather than recognition.
- Dorsal classification has no true depth input.
- Product integration must independently enforce safe action semantics, stop
  behavior, rate limits, and recovery controls.
- Artifact hashes and the exact feature/class order are checked before trusted
  production inference.
- The localhost deployment is intended for a trusted PC; do not expose its ports
  directly to an untrusted network.
