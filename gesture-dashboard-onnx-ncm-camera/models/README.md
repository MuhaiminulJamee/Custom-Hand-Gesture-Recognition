# Eight-Gesture Runtime Artifacts

This directory is an atomic ONNX deployment unit. The trusted base release uses:

- `gesture_mlp_production.onnx` — immutable eight-output classifier plus
  known-gesture-mass output;
- `gesture_mlp_onnx_metadata.json` — hash, exact 76-D input order, canonical
  output order, parity, and offline qualification evidence;
- `gesture_mobile_runtime_config.json` — 20 FPS runtime, action mapping,
  unmirrored coordinates, open-set/temporal gates, low-light policy, landmark
  smoothing, and Safe Learn configuration;
- `hand_landmarker.task` — MediaPipe hand detector/tracker;
- `gesture_online_replay_cache.npz` and
  `gesture_online_validation_cache.npz` — guarded-update evaluation data;
- `gesture_online_untouched_test_cache.npz` — preserved offline release test
  data; never use it to tune a candidate update;
- `eight_gesture_test_metrics.csv`, `eight_gesture_classification_report.csv`,
  `eight_gesture_confusion_matrix.csv`, and
  `eight_gesture_hard_case_metrics.csv` — human-readable exports from the
  preserved eight-class offline evaluation and the reported Rock/Down/Dorsal
  hard cases;
- `gesture_artifact_manifest.json` — exact trusted file sizes and hashes.

The canonical classifier order is:

```text
left, right, up, down, open_palm, like, dorsal, ok
```

Their actions are Move Left, Move Right, Move Up, Move Down, Enable / Disable
Object Tracking, Play / Pause, Return to Default Position, and Start / Stop
Recording. `no_gesture` is the feedback/rejection sentinel; it is not a ninth
classifier probability.

Do not add a Joblib or TFLite fallback. The application fails closed when the
signed ONNX contract is incomplete, reordered, or changed. Camera pixels and
landmarks must remain unmirrored so Left and Right preserve source coordinates.

## Safe Learn state

Safe Learn keeps the ONNX graph unchanged and stores a bounded local NumPy
adapter separately. Its state and backup directory are runtime/user data, not
automatically trusted base artifacts. A clean handover should omit local adapter
state and feedback unless they have been explicitly reviewed and qualified.
Forced or unreviewed learning is unsupported. A reviewed `no_gesture` correction
creates a local rejection prototype rather than changing the eight-output class
order.

## Releasing a replacement base model

Build, evaluate, and inspect the model before rebuilding the manifest. Then run:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_artifact_manifest.py
.\.venv\Scripts\python.exe .\scripts\onnx_preflight.py
```

The manifest is an integrity mechanism, not an accuracy test. The preflight
checks the exact eight-class/76-D/20 FPS contract and loads ONNX Runtime, but it
does not exercise the physical board camera.

The current metadata reports preserved offline accuracy and open-set results.
Treat that JSON and the `eight_gesture_*` exports as release evidence sources;
unrelated historical CSV files are not substitutes for the current metadata and
untouched test cache.

After automated checks pass, separately test the actual NCM stream in normal and
low light at approximately 20 FPS. Verify unmirrored directions, landmark
stability, all eight per-class results, Dorsal/Down confusion, empty-frame false
activations, action debounce, and transport counters before hardware acceptance.
