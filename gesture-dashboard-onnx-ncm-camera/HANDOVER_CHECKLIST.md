# Eight-Gesture ONNX + USB-NCM Camera Handover Checklist

## Release package

- Confirm `models/gesture_mlp_production.onnx`, metadata, runtime configuration,
  MediaPipe task, online-learning evaluation caches, and artifact manifest are
  present.
- Confirm the runtime output order is exactly `left`, `right`, `up`, `down`,
  `open_palm`, `like`, `dorsal`, `ok`.
- Confirm `no_gesture` appears only as the rejection/feedback sentinel, never as
  a ninth probability or command action.
- Confirm the mapped actions are Move Left, Move Right, Move Up, Move Down,
  Enable / Disable Object Tracking, Play / Pause, Return to Default Position,
  and Start / Stop Recording, in that class order.
- Run `verify_ncm_onnx_release.bat` and retain its output.
- Start the app and verify HTTP 200 on ports 3200 and 8200.
- Stop the application before creating the ZIP.
- Do not include `.venv`, `node_modules`, `.runtime`, captured feedback, a local
  online-adapter state/backups, credentials, or secrets in a clean baseline
  handover unless their inclusion was explicitly reviewed.

## Automated software qualification

- Confirm artifact integrity and the release fingerprint pass.
- Confirm the ONNX session exposes one `[N, 76]` float32 input, eight
  probabilities in the canonical order, and one `[N, 1]` known-mass output.
- Confirm the runtime target is 10 FPS with a 100 ms frame budget, even when the
  board camera delivers frames faster.
- Run the exact-class/open-set accuracy test and record its result.
- Run camera-stability tests covering unmirrored pixels, low-light enhancement,
  near-blank rejection, VIDEO timestamps, landmark EMA, jump rejection, and
  reacquisition.
- Run safe-online-learning tests covering validation rejection, atomic state,
  backups/rollback, negative `no_gesture` prototypes, and rejection of forced or
  unreviewed updates.
- Run TypeScript, ESLint, and the production frontend build.

Current preserved offline evidence is 99.15% known-class accuracy, 98.93% macro
F1, 97.47% minimum per-class F1, and 1.83% unknown false acceptance on 1,769
known plus 1,146 unknown samples. ONNX parity is 100% on 256 preserved samples,
with maximum absolute probability error about `1.05e-7`. These are offline
software results only—not live NCM-camera recognition accuracy.

## Required physical NCM acceptance

Hardware acceptance is separate and mandatory:

1. Connect the USB-NCM board and configure its Windows adapter as
   `192.168.50.1/30`.
2. Run `check_ncm_link.bat`; retain the host, adapter, ARP/neighbor, and TCP
   results.
3. Start the dashboard, run UDP discovery, connect the camera, and confirm a
   rising frame count. Camera transport FPS may exceed the inference limit.
4. Confirm the preview and landmarks are unmirrored while inference remains
   capped at 10 FPS. For the front-facing NCM calibration, an index finger
   pointing toward decreasing image x must produce Right; increasing image x
   must produce Left. Confirm those semantic labels still dispatch Move Right
   and Move Left, respectively.
5. Collect a labeled live test for all eight gestures in normal and low light,
   across varied backgrounds, distances, skin tones, hand sizes, and finger
   thicknesses. Use both hands if both are within product scope.
6. Explicitly challenge Left versus Right, Dorsal versus Down, Down versus
   non-command hand shapes, OK versus Open Palm, partial hands, motion blur, and
   empty scenes.
7. Record the live confusion matrix, per-class precision/recall/F1, false
   actions per minute with no command shown, action latency, landmark jitter,
   observed FPS, and frame-budget pass rate. Do not substitute offline metrics.
8. Confirm CRC errors, header errors, invalid JPEG frames, and sequence gaps stay
   zero during a normal run; document any reconnects.
9. In Feedback, confirm the NCM preview and landmarks remain live. Submit one
   reviewed Safe Learn correction and verify that an accepted update is backed
   up—or that a rejected candidate leaves active behavior unchanged. Also verify
   Save only performs no online update.
10. Obtain the product owner's explicit acceptance of the measured live results;
    no live accuracy threshold should be invented after testing.

If the board does not answer ARP/UDP, accept TCP 5000, or provide valid JLIP/JPEG
frames, stop the recognition test and return the NCM transport issue to the
firmware owner. Model/software qualification cannot clear a blocked hardware
path.

## Receiver setup

1. Extract the folder to a normal writable location.
2. Install 64-bit Python 3.11/3.12, Node.js 22+, and the Microsoft Visual C++
   2015–2022 x64 Redistributable.
3. Connect and configure the USB-NCM interface as above.
4. Run `setup_ncm_onnx_dashboard.bat` once with internet access.
5. Run `check_ncm_link.bat`, then `start_ncm_onnx_dashboard.bat`.
6. Open <http://127.0.0.1:3200>, run UDP discovery, connect the board camera, and
   complete the physical acceptance procedure.
