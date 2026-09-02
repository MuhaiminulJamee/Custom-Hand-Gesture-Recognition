# Custom Hand Gesture Recognition

This repository is organized as a monorepo for multiple deployment editions of
the MediaPipe hand-gesture recognition system.

## Editions

| Folder | Runtime | Status |
|---|---|---|
| [`gesture-dashboard-onnx`](gesture-dashboard-onnx/) | MediaPipe + ONNX Runtime | Available |
| [`gesture-dashboard-joblib`](gesture-dashboard-joblib/) | MediaPipe + Python/Joblib | Available |

Each edition is self-contained and has its own setup instructions, dependencies,
runtime artifacts, tests, and documentation.

## Run the ONNX edition on Windows

```powershell
cd gesture-dashboard-onnx
.\setup_onnx_dashboard.bat
.\start_onnx_dashboard.bat
```

Then open <http://127.0.0.1:3100>.

See [`gesture-dashboard-onnx/README.md`](gesture-dashboard-onnx/README.md) for the
complete environment setup and operating instructions.

## Run the Joblib edition on Windows

```powershell
cd gesture-dashboard-joblib
.\setup_dashboard.bat
.\start_dashboard.bat
```

Then open <http://127.0.0.1:3000>.

The Joblib edition includes the trusted SVM, MLP, and OnlineMLP classifiers,
MediaPipe hand-landmark extraction, the exact 76-D feature pipeline, temporal EMA,
geometry resolvers, Follow Object, online learning, production qualification, and
the 10 FPS dashboard. It intentionally contains no ONNX or TFLite gesture
classifier.

See [`gesture-dashboard-joblib/README.md`](gesture-dashboard-joblib/README.md) for
complete setup, model-safety, testing, and production instructions.
