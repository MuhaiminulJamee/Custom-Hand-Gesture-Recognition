# Custom Hand Gesture Recognition

This repository is organized as a monorepo for multiple deployment editions of
the MediaPipe hand-gesture recognition system.

## Editions

| Folder | Runtime | Status |
|---|---|---|
| [`gesture-dashboard-onnx`](gesture-dashboard-onnx/) | MediaPipe + ONNX Runtime | Available |
| `gesture-dashboard-joblib` | MediaPipe + Python/Joblib | Planned |

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
