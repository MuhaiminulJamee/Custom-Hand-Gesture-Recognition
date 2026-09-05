# Gesture Dashboard — ONNX + USB-NCM Board Camera

This is the isolated development-board-camera edition of the gesture system. It
keeps the same MediaPipe 76-D feature pipeline, ONNX classifier, EMA stability,
gesture resolvers, Follow Object procedure, action demo, feedback capture, and
10 FPS measurements as the standard ONNX dashboard. Only the live camera input
has changed.

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
MediaPipe Hand Landmarker → normalized 76-D features → ONNX Runtime MLP
        │
        ▼
EMA/stability gate → geometry resolvers → mapped actions / Follow Object
        │
        ▼
Dashboard preview, landmarks, probabilities, timing, uploaded media actions
```

The model does not need retraining for this camera integration. Camera-domain
validation is still required because the board lens, exposure, orientation, and
image quality may differ from the original training/camera environment.

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
# Optional, only if the board image orientation requires it:
$env:NCM_ROTATE_180 = "false"
$env:NCM_MIRROR_HORIZONTAL = "false"
.\start_ncm_onnx_dashboard.bat
```

The start script supplies the defaults when variables are not set.

## Verification

Run the complete release checks:

```powershell
.\verify_ncm_onnx_release.bat
```

The JLIP parser tests cover fragmented packets, coalesced packets, stream
resynchronization, CRC rejection, maximum payload enforcement, and network byte
order. A final end-to-end pass still requires the physical board and corrected
firmware/network services.

## Model and security notes

- `gesture_mlp_production.onnx` is inference-only; user feedback is saved for a
  later reviewed offline retraining/export cycle.
- Artifact hashes are checked before inference in production mode.
- The API binds only to `127.0.0.1`; the board connection is outbound from the
  backend over the dedicated NCM subnet.
- `no_gesture` quality remains the separately documented deferred model task.
