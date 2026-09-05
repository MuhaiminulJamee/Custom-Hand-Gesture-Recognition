# ONNX + USB-NCM Camera Handover Checklist

Before sending the package:

- Confirm `models/gesture_mlp_production.onnx` and its metadata are present.
- Run `verify_ncm_onnx_release.bat`.
- Start the app and verify HTTP 200 on ports 3200 and 8200.
- Run `check_ncm_link.bat` and save the host/ARP/TCP result.
- Test UDP discovery, a live JLIP frame stream, and at least one uploaded
  image/video action target.
- Confirm CRC errors, header errors, invalid JPEG frames, and sequence gaps remain
  zero during a normal test.
- Confirm the Analytics tab reports `ONNX` as the runtime model.
- Stop the application before creating the ZIP.
- Do not include `.venv`, `node_modules`, `.runtime`, captured feedback, or
  secrets.

The receiver should:

1. Extract the folder to a normal writable location.
2. Install 64-bit Python 3.11/3.12, Node.js 22+, and the Microsoft Visual C++
   2015–2022 x64 Redistributable.
3. Connect the USB-NCM development board and configure its Windows adapter as
   `192.168.50.1/30`.
4. Run `setup_ncm_onnx_dashboard.bat` once with internet access.
5. Run `check_ncm_link.bat`, then `start_ncm_onnx_dashboard.bat`.
6. Open http://127.0.0.1:3200, run UDP discovery, and connect the board camera.

Known model limitation: `no_gesture` remains the explicitly deferred class for
future enrichment. The ONNX conversion itself has 100% prediction agreement with
the selected source MLP on the 256-sample conversion validation subset.

Hardware acceptance is separate from software verification. If the board does
not answer ARP/UDP or accept TCP 5000, the firmware owner must resolve the NCM
transport before an end-to-end gesture test is possible.
