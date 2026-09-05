# Eight-Gesture Dashboard — ONNX + USB-NCM Board Camera

Current release: **v18_20**. See [V18_20_README.md](V18_20_README.md) for current training data, notebook, metrics, and limitations. Prior-release statistics below are historical.

This is the isolated development-board-camera edition of the gesture system. It
uses an eight-output ONNX classifier, MediaPipe's 76-D hand feature pipeline,
open-set rejection, geometry checks, temporal stability, safe reviewed online
learning, and a 10 FPS application target. Frames come from the development
board rather than a browser webcam.

The browser does **not** request webcam permission. The Python backend connects
directly to the board at `192.168.50.2:5000`, reassembles the JLIP/TCP stream,
validates each packet, and sends its JPEG frames into the existing inference
pipeline.

This folder is independent from:

- `gesture-dashboard-onnx` — browser webcam, ports 3100/8100;
- `gesture-dashboard-joblib` — Joblib runtime, ports 3000/8000.

This edition uses ports **3200** and **8200**, so all three can be installed and
run separately.

## Architecture

```text
Development-board camera
        │ JPEG frames
        ▼
USB-NCM network link (Windows 192.168.50.1/30 ↔ board 192.168.50.2)
        │ TCP 5000, JLIP v1 packets
        ▼
NCM camera client
  CRC32 · sequence tracking · reconnect · JPEG validation
        │
        ▼
low-light/blank-frame checks → MediaPipe VIDEO Hand Landmarker
        │
        ▼
smoothed landmarks → normalized 76-D features → eight-output ONNX Runtime MLP
        │
        ▼
known-mass + geometry + probability-margin + EMA/stability rejection
        │
        ▼
eight mapped actions, feedback capture, and optional safe local adaptation
```

The release model was rebuilt to expose only the eight approved command labels.
Offline qualification is strong, but it does not prove live board-camera
performance. Camera-domain validation is still required because the board lens,
exposure, frame delivery, backgrounds, hand appearance, and image quality may
differ from the preserved evaluation data.

## Eight-gesture contract

Classifier output order and action mapping are fixed:

| Label | Display name | Action |
|---|---|---|
| `left` | Left | Move Left |
| `right` | Right | Move Right |
| `up` | Up | Move Up |
| `down` | Down | Move Down |
| `open_palm` | Open Palm | Enable / Disable Object Tracking |
| `like` | Like | Play / Pause |
| `dorsal` | Dorsal | Return to Default Position |
| `ok` | OK | Start / Stop Recording |

`no_gesture` is a rejection and feedback label, not a ninth ONNX probability.
It is emitted when no usable hand is present or when known-class confidence,
probability margin, pose geometry, or temporal stability is insufficient. In
the Feedback tab, choose it to mark a false detection or background frame.

The preview and inference pixels remain unmirrored. NCM horizontal calibration
swaps the legacy source-camera Left/Right interpretation: decreasing image x is
reported as Right and increasing image x is reported as Left, matching the
front-facing operator's intended direction. The emitted semantic labels still
map normally (`left` to Move Left and `right` to Move Right); no pixel flip is
performed in the preview, landmarks, feature extraction, or action demo.

## Live stability and low-light handling

The live pipeline uses a backend-wide hard cap of 10 inference starts per second,
even if the NCM camera delivers frames faster or multiple clients are connected.
Failed attempts also consume their slot. Its 100 ms frame budget leaves backend
headroom while preserving the following stability controls:

- MediaPipe runs in VIDEO mode with monotonic timestamps so its tracker can use
  frame-to-frame continuity.
- Adaptive luminance analysis applies graded CLAHE/gamma enhancement only to
  underexposed frames. Near-blank frames are rejected before hand inference.
- Invalid, implausibly small, clipped, or isolated jumping landmark sets are
  rejected. Accepted landmarks use velocity-adaptive EMA smoothing, with
  persistent-jump reacquisition instead of following one-frame spikes.
- Directional, Dorsal, Like, OK, and Open Palm pose checks veto incompatible
  geometry before an action can fire.
- The classifier uses known-gesture mass, a 0.80 confidence floor, a 0.18
  probability-margin floor, three stable frames, three release frames, and an
  action cooldown. These gates intentionally prefer no action to a false action.

These controls reduce jitter and false positives; they do not replace testing
with the physical camera in the intended lighting and mounting conditions.

## Board protocol implemented

The implementation follows the supplied `NcmCameraTester.cpp`, preserved at
[`reference/NcmCameraTester.cpp`](reference/NcmCameraTester.cpp).

- Windows host address: `192.168.50.1`
- board address: `192.168.50.2`
- JPEG stream: TCP port `5000`
- discovery: UDP port `5001`
- discovery payload: ASCII `JL-CAMERA-DISCOVER`
- JLIP header: 24 bytes, network byte order
- magic/version: `JLIP`, version `1`
- packet types: Hello ACK `0x02`, heartbeat `0x03`, JPEG `0x10`
- packet checks: declared length, 200 KiB maximum payload, CRC32, JPEG markers

TCP is a byte stream. One JLIP frame may be split across many TCP reads or
several packets may arrive in one read. The parser handles both cases and
resynchronizes after invalid bytes. A 2048-byte USB NTB limit does **not** limit a
JPEG to 2048 bytes; firmware and TCP must fragment/reassemble it correctly.

## Requirements

- 64-bit Windows 10 or 11
- 64-bit Python 3.11 or 3.12
- Node.js 22.13 or newer
- Corepack/pnpm
- Microsoft Visual C++ 2015–2022 x64 Redistributable
- a working USB-NCM driver/interface for the development board
- board firmware providing UDP discovery on 5001 and JLIP TCP streaming on 5000

## First-time setup in VS Code

Open this folder directly in VS Code:

```powershell
cd "C:\path\to\Custom-Hand-Gesture-Recognition\gesture-dashboard-onnx-ncm-camera"
code .
```

In the VS Code PowerShell terminal, run:

```powershell
.\setup_ncm_onnx_dashboard.bat
```

The setup creates this edition's own `.venv`, installs frontend dependencies,
checks ONNX Runtime, downloads the MediaPipe landmarker only if missing, verifies
the model artifacts, and runs the ONNX preflight.

The checked-in runtime does not need training libraries. To reproduce the
eight-output ONNX build and its CSV/cache evidence from the preserved source
artifacts, install `requirements-model-build.txt` and run
`scripts/build_eight_gesture_model.py`; rebuild the artifact manifest only after
the generated files have been reviewed.

## Configure and verify the NCM interface

The Windows NCM adapter must own `192.168.50.1` with prefix length `30`. Do not
assign that address to Wi-Fi or Ethernet; select the USB-NCM adapter exposed by
the development board.

Inspect the current configuration:

```powershell
Get-NetAdapter
Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object IPAddress -eq "192.168.50.1"
```

Then run the included read-only diagnostic:

```powershell
.\check_ncm_link.bat
```

It reports the host address, prefix length, ARP/neighbor state, source adapter,
and whether TCP port 5000 accepts a connection. It does not modify Windows
network settings.

## Start and stop

```powershell
.\start_ncm_onnx_dashboard.bat
```

Open:

- dashboard: <http://127.0.0.1:3200>
- API documentation: <http://127.0.0.1:8200/docs>
- health/NCM status: <http://127.0.0.1:8200/api/health>

In the dashboard:

1. Select **Run UDP discovery**.
2. Confirm that the board replies from `192.168.50.2:5001`.
3. Select **Connect board camera**.
4. Wait for state `Connected` and a rising frame count.
5. Show a gesture and verify predictions, landmarks, EMA stability, actions, and
   the 10 FPS capacity panel.
6. Optionally upload a video of up to 60 seconds or an image and demonstrate the
   mapped gesture actions against it.

Stop only this edition with:

```powershell
.\stop_ncm_onnx_dashboard.bat
```

## NCM-specific API

| Endpoint | Purpose |
|---|---|
| `GET /api/ncm/status` | Connection, frame rate, counters, and latest error |
| `POST /api/ncm/discover` | Send the UDP discovery payload and await a reply |
| `POST /api/ncm/connect` | Start the reconnecting TCP/JLIP camera client |
| `POST /api/ncm/disconnect` | Stop the board-camera client |
| `GET /api/ncm/frame.jpg` | Return the latest validated JPEG |
| `GET /api/ncm/stream.mjpg` | Browser preview of validated board frames |
| `WS /ws/ncm-live` | Server-pushed predictions from board frames |

## Reviewed feedback and Safe Learn

The Feedback tab keeps the NCM preview and landmark overlay live while the user
confirms the actual label. **Safe Learn** is the default; **Save only** records
the sample without changing live behavior.

Safe Learn does not rewrite the immutable ONNX file. It proposes a bounded local
NumPy residual-adapter update from one explicitly reviewed 76-D sample, evaluates
the candidate against validation/replay data and open-set guardrails, and accepts
it only when the safety gates pass. Accepted state is saved atomically with a
backup; rejected candidates leave the active adapter unchanged. A reviewed
`no_gesture` sample teaches a local rejection region instead of creating another
classifier class. Offline retraining and a fresh ONNX export remain the path for
a new base-model release.

## If discovery and TCP currently fail

The reported iPad state (`en3`, path satisfied, `192.168.50.1/30`) only proves
that the local NCM interface exists. It does not prove that the board answers
ARP or that its UDP/TCP services are running. If discovery receives no reply and
TCP SYN never completes, gesture inference cannot begin yet.

The firmware developer should check, in this order:

1. Confirm iPad/Windows ARP requests reach `usb_ncm_parse_ntb()`.
2. Confirm firmware sends a correct ARP reply for `192.168.50.2`.
3. Log and validate NTH16/NDP16 signatures, `wBlockLength`, and `wNdpIndex`.
4. Accept multiple datagrams in one iOS NTB and enforce the required 4-byte
   datagram alignment.
5. Inspect the USB RX-drop counter and the 2048-byte NTB handling. Never assume
   one NTB equals one Ethernet, IP, TCP, JLIP, or JPEG message.
6. Confirm `recvfrom(5001)` receives `JL-CAMERA-DISCOVER` and send a UDP reply.
7. Confirm the firmware listener is bound and `accept(5000)` receives the TCP
   connection.
8. After TCP connects, confirm the board sends valid JLIP headers and complete
   JPEG payloads with matching CRC32 values.

The Windows application exposes CRC errors, invalid headers/JPEGs, sequence
gaps, reconnects, and the last socket error. It cannot inspect
`usb_ncm_parse_ntb()`, NTH/NDP fields, firmware RX counters, `recvfrom`, or
`accept`; those checks must be instrumented in the board firmware.

## Configuration

Defaults are defined in `.env.example` and can be overridden before startup:

```powershell
$env:NCM_HOST_IP = "192.168.50.1"
$env:NCM_DEVICE_IP = "192.168.50.2"
$env:NCM_TCP_PORT = "5000"
$env:NCM_DISCOVERY_PORT = "5001"
# Rotation may be used only when the sensor is physically upside down.
$env:NCM_ROTATE_180 = "false"
.\start_ncm_onnx_dashboard.bat
```

The start script supplies the defaults when variables are not set. Horizontal
mirroring remains deliberately unsupported. Left/Right are corrected by the NCM
semantic calibration, not by modifying camera pixels.

## Verification

Run the complete release checks:

```powershell
.\verify_ncm_onnx_release.bat
```

The JLIP parser tests cover fragmented packets, coalesced packets, stream
resynchronization, CRC rejection, maximum payload enforcement, and network byte
order. The automated model tests cover the exact eight-output contract,
unmirrored pixels, calibrated Left/Right semantics, preserved offline
accuracy/open-set gates,
low-light preprocessing, blank-frame suppression, landmark smoothing/jump
handling, and guarded online learning.

The pre-v18_20 preserved offline evaluation reported 1,769 known-class samples and
1,146 held-out unknown samples: 99.15% known-class accuracy, 98.93% macro F1,
97.47% minimum per-class F1, and 1.83% unknown false acceptance at the configured
known-mass threshold. ONNX conversion parity is 100% on 256 preserved samples
with maximum absolute probability error about `1.05e-7`. These are offline
artifact results, not live NCM-camera accuracy claims.

A release acceptance pass still requires the physical NCM board. Test all eight
gestures in normal and low light, with varied backgrounds, distances, left/right
hands, skin tones, hand sizes, and finger thicknesses. Record per-class results
and specifically challenge Left versus Right, Dorsal versus Down, Down versus
non-command hand shapes, and empty-frame false positives. Confirm sustained
approximately 10 FPS inference, a 100 ms processing budget, stable landmarks,
action debounce, and zero transport errors during the run. The camera transport
may report more than 10 FPS; the backend must still cap inference at 10 FPS.

## Model and security notes

- `gesture_mlp_production.onnx` remains immutable. Safe Learn updates only the
  bounded, validation-gated local adapter; Save only retains feedback for offline
  retraining and a qualified future ONNX export.
- Artifact hashes are checked before inference in production mode.
- The API binds only to `127.0.0.1`; the board connection is outbound from the
  backend over the dedicated NCM subnet.
- `no_gesture` is a feedback/rejection sentinel and is never exposed as a ninth
  command probability or mapped action.
