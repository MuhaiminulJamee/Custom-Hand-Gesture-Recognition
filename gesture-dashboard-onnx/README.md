# Gesture Control Lab — Independent ONNX Runtime Edition

This folder is a standalone Windows deployment of the qualified gesture MLP in
ONNX format. It does **not** load or require Joblib, scikit-learn, TensorFlow,
TFLite, the notebook, Google Drive, or the original dashboard directory.

The package keeps the application behavior around the classifier:

- MediaPipe hand detection and the exact 76-D landmark feature contract;
- 15 gesture probabilities and geometry-based directional/zoom resolution;
- EMA confidence/stability gating and mapped actions;
- continuous-camera Palm → Fist → Palm Follow Object procedure;
- a local uploaded-video/image action demo with live webcam and landmark picture-in-picture;
- 10 FPS frame-budget measurement and runtime observability;
- image upload testing, action history, and reviewed feedback capture.

The ONNX model is immutable. Feedback is stored for later offline retraining and
re-export; it is never presented as in-place ONNX learning.

## Included runtime artifacts

The `models` folder already contains:

- `gesture_mlp_production.onnx` — qualified classifier;
- `gesture_mlp_onnx_metadata.json` — feature order, class order, SHA-256, opsets,
  and conversion-parity evidence;
- `gesture_mobile_runtime_config.json` — thresholds, mappings, and timing rules;
- `hand_landmarker.task` — MediaPipe hand detector;
- `gesture_artifact_manifest.json` — deployment integrity manifest.

The ONNX conversion was checked on 256 preserved validation samples with 100%
prediction agreement and a maximum absolute probability error of approximately
`1.31e-6` against the selected Python MLP.

## Windows requirements

Install these once on the target PC:

1. 64-bit Windows 10 or 11.
2. 64-bit Python 3.11 or 3.12, with the Python launcher added to PATH.
3. Node.js 22 or newer (the setup script uses `pnpm` through Corepack when
   `pnpm` is not already installed).
4. [Microsoft Visual C++ 2015–2022 Redistributable (x64)](https://aka.ms/vc14/vc_redist.x64.exe).
5. A laptop camera, USB webcam, or phone configured as a Windows webcam.

The Microsoft runtime is required by the Windows ONNX Runtime native DLL. If it
is missing, setup stops with a clear message instead of starting a broken app.

## First-time setup

Open this folder in File Explorer and double-click:

```text
setup_onnx_dashboard.bat
```

It creates independent `.venv` and `node_modules` directories, installs only the
pinned ONNX-runtime application dependencies, verifies artifact hashes, opens an
ONNX Runtime CPU session, checks the 76 → 15 contract, and initializes MediaPipe.

Equivalent VS Code terminal command:

```powershell
cd "C:\path\to\gesture-dashboard-onnx"
.\setup_onnx_dashboard.bat
```

An internet connection is needed during this first setup. The model and
MediaPipe task are already included, so no training or Drive download is needed.

## Start and stop

Start the independent ONNX application:

```powershell
.\start_onnx_dashboard.bat
```

Then open:

- Dashboard: http://127.0.0.1:3100
- API documentation: http://127.0.0.1:8100/docs
- Health: http://127.0.0.1:8100/api/health

Stop only the ONNX application:

```powershell
.\stop_onnx_dashboard.bat
```

Do not close the startup terminal until it reports that both services are ready.
The actual web/backend processes run in the background and can be stopped with
the stop script.

## Run Joblib and ONNX side by side

The two folders and environments are independent:

| Edition | Dashboard | API | Start command |
|---|---:|---:|---|
| Existing Joblib | `127.0.0.1:3000` | `127.0.0.1:8000` | `gesture-dashboard\start_dashboard.bat` |
| Independent ONNX | `127.0.0.1:3100` | `127.0.0.1:8100` | `gesture-dashboard-onnx\start_onnx_dashboard.bat` |

They can run simultaneously because they use different folders, environments,
PID files, logs, feedback folders, and ports.

## Use the live test

1. Open the ONNX dashboard and confirm the Setup tab reports **READY**.
2. Choose **Start webcam** and allow camera permission in the browser.
3. Keep one complete hand and wrist within the guide.
4. Hold the pose until the configured confidence and stable-frame gates pass.
5. Review the runtime prediction, mapped label, action, probabilities, and 10 FPS
   verdict.
6. For Follow Object, press **Begin Follow Object** and perform Palm → Fist →
   Palm in one continuous camera session.

## Demonstrate real gesture actions on a video or image

The Live tab includes an **Upload video / image** control. The file stays inside
the browser; videos must be no longer than 60 seconds. After loading it, start the
webcam. The uploaded media appears inside a smaller framed action area, leaving
visible space around every edge so Move Up/Down/Left/Right and zoom operations are
easy to see. The mirror-corrected webcam feed,
MediaPipe hand skeleton, current gesture, confidence, and confirmed action appear
in the upper-right picture-in-picture view.

Only a stable, confidence-gated runtime action is dispatched. Holding the same
pose does not repeatedly execute it; release or change the gesture before using
that command again. Every performed command produces a short glowing confirmation
over the video (for example, **MOVED UP**, **VIDEO PAUSED**, or **ZOOMED IN**), and
the on-screen action-dispatch log keeps the recent evidence.

| Gesture | Demonstration operation |
|---|---|
| `call` | Activates the visible face-target command marker. This proves dispatch; it is not a second face-recognition model. |
| `rock` | Pauses the uploaded video. |
| `like` | Toggles video playback between stop and continue. |
| `ok` | Starts a real browser recording of the uploaded video stream. |
| `peace` | Ends recording and exposes a downloadable WebM file. |
| `one` / `one_down` | Moves the target video up / down. |
| `one_left` / `one_right` | Moves the target video left / right. |
| `zoom_in` / `zoom_out` | Increases / decreases the target zoom. |
| `dorsal_hand` | Returns position and zoom to the main state. |
| `fist` | Marks the object as grabbed for the Follow Object demonstration. |
| `palm` | Releases a grabbed object; otherwise opens/plays the video. |
| `Palm → Fist → Palm` | Completes the dedicated Follow Object procedure. |
| `no_gesture` | Performs no operation and re-arms the gesture dispatcher. |

The recording feature is intended for current Chrome or Edge on Windows and uses
the browser `MediaRecorder` and canvas-capture APIs. It records demonstrated
movement for either a video or an image. Webcam inference frames are processed by
the local backend; uploaded media remains local to the browser.

The browser intentionally caps requests at 10 FPS. A `PASS` means the measured
MediaPipe + feature extraction + ONNX inference + application work fits within
the 100 ms frame budget on that PC.

## Direct ONNX Runtime example

To test the model without the dashboard, provide a JSON file containing one
finite 76-number feature vector:

```powershell
.\.venv\Scripts\python.exe .\examples\onnx_inference_example.py `
  --features-json C:\path\to\features.json
```

The example validates the same metadata and model hash used by the application,
then prints all 15 probabilities and the predicted label.

## Feedback behavior

The Feedback tab saves reviewer corrections under `feedback` with the snapshot,
landmarks, 76-D vector, class probabilities, and note. To incorporate those
samples, retrain outside this deployment, re-export ONNX, rerun conversion parity,
replace the model and metadata, and rebuild the manifest. The deployed ONNX graph
does not modify itself.

## Handover package

Before packaging, double-click `verify_onnx_release.bat` to repeat the dependency,
test, preflight, type, lint, and frontend-build checks.

Run:

```powershell
.\scripts\create_handover_zip.ps1
```

This creates a timestamped ZIP beside this folder. It includes source code,
scripts, pinned dependency files, documentation, and model artifacts, while
excluding machine-specific `.venv`, `node_modules`, runtime logs, caches, and
captured feedback. Your boss extracts the ZIP and runs the setup script once.

See [HANDOVER_CHECKLIST.md](HANDOVER_CHECKLIST.md) and [MODEL_CARD.md](MODEL_CARD.md)
before distribution.

## Troubleshooting

### `DLL load failed while importing onnxruntime_pybind11_state`

Install the Microsoft Visual C++ x64 Redistributable linked above, restart the
terminal or PC, and rerun `setup_onnx_dashboard.bat`.

### `127.0.0.1 refused to connect`

Run `start_onnx_dashboard.bat`. If startup fails, inspect:

```text
.runtime\backend.stderr.log
.runtime\backend.stdout.log
.runtime\frontend.stderr.log
.runtime\frontend.stdout.log
```

### Port 3100 or 8100 is busy

Run `stop_onnx_dashboard.bat`. If another application owns a port, stop that
application or change both the PowerShell start script and the frontend/API URL
consistently.

### Browser webcam unavailable

Allow camera permission in the browser and Windows privacy settings, and close
other applications using that webcam.

### Replacing the model

Do not replace only the `.onnx` file. The model, metadata, runtime configuration,
class order, and feature order are one release unit. A changed file makes the
integrity preflight fail by design.
