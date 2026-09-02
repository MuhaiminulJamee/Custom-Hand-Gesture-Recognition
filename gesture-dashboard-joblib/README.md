# Gesture Control Lab — Joblib Edition

A Joblib-only Windows application for testing whether the **v18_17 gesture model can
sustain a strict 10 FPS camera workload**. The browser supplies the webcam and a
polished dashboard; a local Python service performs MediaPipe landmark detection,
the exact 76-D feature extraction, model inference, geometry resolution, temporal
EMA filtering, Follow Object, feedback logging, and optional guarded online
learning.

The application is private to the PC by default. Both services listen only on
`127.0.0.1`.

## What is implemented

- The exact 15-class v18_17 order:
  `call`, `rock`, `like`, `ok`, `one`, `one_down`, `one_left`, `one_right`,
  `palm`, `peace`, `dorsal_hand`, `fist`, `zoom_in`, `zoom_out`, `no_gesture`.
- MediaPipe Hand Landmarker with one-hand processing and the notebook's 76-D
  landmark/shape feature contract.
- SVM, MLP, and OnlineMLP loading from trusted Joblib artifacts.
- Directional `one` resolution, zoom resolution, Return Main geometry, confidence
  gating, EMA smoothing, stable-frame gating, zoom hold, and command cooldown.
- Dedicated, opt-in Follow Object with one continuous camera:
  `palm → HaGRID fist → palm`. The RPS `rock` class remains Pause Video and cannot
  substitute for `fist`.
- Follow Object instructions inside the camera frame, a large camera-off success
  message, and automatic success dismissal after eight seconds.
- Strict back-pressure capture: only one frame can be in flight. The next frame is
  scheduled so completed inference is capped at 10 FPS without building a queue.
- Per-model diagnostics, all class probabilities, action history, model artifact
  checks, uploaded-image testing, and reviewed feedback.
- Audit-only feedback and validation-gated Safe Learn. Force Learn exists only as
  an explicitly enabled, authenticated maintenance operation and is disabled by
  the normal production launcher.
- SHA-256 integrity verification before trusted Joblib artifacts are loaded,
  startup preflight, strict model selection, bounded uploads/sessions, rotating
  structured logs, rollback checkpoints, and frozen release qualification.
- Rolling p50/p95/p99 pipeline latency, frame-budget pass rate, error rate,
  liveness/readiness endpoints, and Prometheus-compatible metrics.

## Included trained classifiers

The trusted Joblib classifier bundle, primary classifier, online checkpoint, and
MediaPipe detector are included in `models/`. **You do not need to retrain on the
PC.** Use the import procedure below only when replacing them with a newer trusted
Joblib export from your own notebook run.

Only load Joblib files created by you. Joblib/pickle files can execute Python code
while loading and must not come from an untrusted source.

## Requirements

- Windows 10 or 11, 64-bit.
- Python 3.11 recommended. During setup, the Python launcher command `py -3.11`
  is preferred when available.
- Node.js 22 or newer, including Corepack, or a working `pnpm` command.
- A webcam. A camera that is physically limited to 10 FPS is supported.
- Your own v18_17 Joblib exports when replacing the included model.

No CUDA GPU is required. The test is designed to measure the CPU on the PC that
will actually run the application.

## First-time setup

### 1. Create the environments

Double-click:

```text
setup_dashboard.bat
```

It creates `.venv`, installs the Python packages, installs the pinned frontend
packages, and verifies/downloads the MediaPipe hand detector. Internet is needed
for this first setup only.

PowerShell equivalent:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

### 2. Download the notebook exports from Drive

Download the files from the same Drive directory used by the v18_17 notebook,
normally the `GestureRecognitionOnline/follow_object_hagrid_fist_v1` output
folder.

For the recommended Python runtime, copy these files:

- `gesture_joblib_runtime_config.json`
- `mediapipe_hagrid_rps_pinch15_hagrid_fist_follow_object_models.joblib`
- `gesture_primary_model.joblib` (optional fallback)
- `gesture_online_state.joblib` (recommended when OnlineMLP is the latest model)

For PC feedback-based online learning, also copy:

- `gesture_online_replay_cache.npz`
- `gesture_online_validation_cache.npz`
- `gesture_online_confirmed_features.npz` (optional existing feedback history)
- `gesture_online_untouched_test_cache.npz` (evaluation archive)

Evaluation CSV files are optional but will appear in the Analytics tab.

### 3. Import the downloaded files

From PowerShell in this project folder, replace the example folder with the folder
you downloaded from Drive:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\import_models.ps1 `
  -SourceDirectory "C:\Users\Admin\Downloads\follow_object_hagrid_fist_v1"
```

The script copies only recognized v18_17 artifacts into `models`, preserves the
previous files in a timestamped import backup, generates the frozen quality
report, and signs the runtime artifacts. Prefer this script over manual copying;
manual changes intentionally make the integrity check fail.

### 4. Start the application

Double-click:

```text
start_dashboard.bat
```

It starts the two background services and opens:

- Dashboard: `http://127.0.0.1:3000`
- API documentation: `http://127.0.0.1:8000/docs`

The launcher builds and serves the optimized production frontend. This uses less
memory than Vinext's three-environment development server and is the recommended
mode for live camera/model testing on Windows.

To close both services, double-click:

```text
stop_dashboard.bat
```

## Running the 10 FPS qualification test

1. Open **Live** and click **Start 10 FPS test**.
2. Allow camera permission when the browser asks.
3. Keep one complete hand and wrist inside the large green guide.
4. Test several classes for at least 20–30 seconds, including `no_gesture`.
5. Read the **10 FPS Capacity Test** verdict and Pipeline Health values.

The measurements mean:

| Metric | Meaning |
| --- | --- |
| Camera FPS | Frames physically delivered by the webcam/browser. A 10 FPS camera should be near 10. |
| Configured FPS limit | The application cap: exactly 10 FPS. |
| Effective application FPS | Frames that completed the local round trip and inference, capped at 10. |
| Total pipeline time | End-to-end Python work for the most recent frame. |
| Uncapped pipeline FPS | `1000 / total pipeline time`; the PC's approximate maximum processing capacity. |
| Frame budget | 100 ms, because `1000 / 10 FPS = 100 ms`. |
| Budget used | `total pipeline time / 100 ms × 100%`. |
| Headroom | `100 ms - total pipeline time`. Positive is good. |

**PASS** means the measured pipeline work fits inside 100 ms, so the PC has enough
processing capacity for a 10 FPS input. For example, 61.66 ms gives about 16.22
uncapped FPS, uses 61.7% of the budget, and therefore passes at the configured 10
FPS. The first few frames are warm-up frames; judge the stable readings.

If the webcam itself supplies only 10 FPS, the model still runs. More camera FPS
does not make this test process more than 10 FPS because the application cap and
one-frame back pressure remain active.

## Follow Object

Click **Begin Follow Object**. This is deliberately opt-in so normal static
commands are untouched.

1. The frame shows **Begin Follow the object procedure** — hold `palm`.
2. It shows **Grab the object** — hold the dedicated HaGRID `fist`.
3. It shows **Release the object** — hold `palm` again.
4. On success the webcam is stopped, then a large **FOLLOW OBJECT DONE
   SUCCESSFULLY** message is shown for eight seconds.

The sequence uses an equal-probability SVM+MLP ensemble when both base models are
available, matching the notebook's drift-resistant Follow Object safeguard.

## Feedback and online learning

After a successful live or uploaded-image prediction, open **Feedback**, choose
the true class, and choose one action:

- **Save feedback only** writes the image, probabilities, landmarks, and 76-D
  features to `feedback/` without changing the classifier.
- **Safe Learn** creates a candidate OnlineMLP update using the new sample plus
  balanced replay. It accepts only when macro-F1 drop is at most `0.003`, maximum
  per-class F1 drop is at most `0.02`, and the confirmed-class probability rises.
- **Force Learn** is disabled by the normal production launcher. It may only be
  enabled for an authenticated, isolated maintenance session and should never be
  offered to ordinary users.

Accepted updates persist to `models/gesture_online_state.joblib`, the model bundle,
and `gesture_primary_model.joblib`; they survive application restarts. In production
they are treated as candidates: the last frozen qualified model remains selected
until `scripts\qualify_release.py` evaluates all SVM/MLP/OnlineMLP candidates and
writes a new production selection. The offline StandardScaler remains frozen and
each update uses eight new-sample repeats plus balanced replay.

This edition updates and runs only Python/Joblib checkpoints. The updated
OnlineMLP remains available as a candidate while the currently qualified Joblib
model stays live until release qualification passes.

## Dashboard sections

- **Live** — webcam, landmarks, current prediction/action, all 15 probabilities,
  Follow Object, and the 10 FPS verdict.
- **Analytics** — SVM/MLP/OnlineMLP diagnostics, timings, recent emitted actions,
  class/action mapping, and notebook CSV exports.
- **Feedback** — reviewed labels, audit capture, Safe Learn, and Force Learn.
- **Setup** — artifact status and the exact runtime contract.

## Developer verification

With setup complete:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\node_modules\.bin\tsc.cmd --noEmit
.\node_modules\.bin\vite.cmd build --config vite.local.config.ts
```

The local API can be checked at `http://127.0.0.1:8000/api/health`. See
`PRODUCTION.md` for release qualification, monitoring, rollback, environment
controls, and incident response.

## Troubleshooting

### “Model setup required”

The model bundle, runtime JSON, or signed manifest is missing. Stop the dashboard,
import the Drive files with `scripts\import_models.ps1`, then restart it. Hot reload
is intentionally disabled by the normal production launcher.

### Runtime JSON says the class order is wrong

The application accepts only the exact v18_17 15-class order. An older 14-class
export is intentionally rejected because its probabilities would be assigned to
the wrong labels after the inserted `fist` class.

### Camera permission denied

Open browser permissions for `127.0.0.1`, allow the camera, and ensure another
program is not holding the webcam.

### Camera FPS is higher than 10

Some webcam drivers ignore the requested device constraint. This is not a test
failure: capture/inference is still strictly limited to 10 completed frames per
second. The dashboard displays Camera FPS and Effective Application FPS separately.

### Effective FPS is below 10

Check Total Pipeline Time. If it is over 100 ms, the PC cannot sustain the full
configured work. Close heavy programs, select the faster Joblib MLP, reduce camera
resolution, or test on the target hardware. If total time is under
100 ms but effective FPS is slightly below 10, allow warm-up and verify that the
camera itself is actually providing 10 FPS.

### Port 3000 or 8000 is already in use

Run `stop_dashboard.bat`, then start again. The stop script only targets processes
previously launched by this dashboard.

### Online learning is unavailable

Import `gesture_online_replay_cache.npz` and
`gesture_online_validation_cache.npz`, plus the OnlineMLP checkpoint or full model
bundle, then restart the dashboard.
