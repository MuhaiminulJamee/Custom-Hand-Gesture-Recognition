# ONNX Handover Checklist

Before sending the package:

- Confirm `models/gesture_mlp_production.onnx` and its metadata are present.
- Run `scripts\onnx_preflight.py` in the independent `.venv`.
- Run the Python tests, TypeScript check, ESLint, and frontend build.
- Start the app and verify HTTP 200 on ports 3100 and 8100.
- Test at least one uploaded image/video action target and one live webcam session.
- Confirm the Analytics tab reports `ONNX` as the runtime model.
- Stop the ONNX application before creating the ZIP.
- Run `scripts\create_handover_zip.ps1`.
- Do not include `.venv`, `node_modules`, `.runtime`, captured feedback, or secrets.

The receiver should:

1. Extract the ZIP to a normal writable folder.
2. Install 64-bit Python 3.11/3.12, Node.js 22+, and the Microsoft Visual C++
   2015–2022 x64 Redistributable.
3. Run `setup_onnx_dashboard.bat` once with internet access.
4. Run `start_onnx_dashboard.bat`.
5. Connect or enable the Windows webcam, open http://127.0.0.1:3100, allow
   browser camera permission, and start the webcam.

Known release limitation: `no_gesture` remains the explicitly deferred class for
future enrichment. The ONNX conversion itself has 100% prediction agreement with
the selected source MLP on the 256-sample conversion validation subset.
