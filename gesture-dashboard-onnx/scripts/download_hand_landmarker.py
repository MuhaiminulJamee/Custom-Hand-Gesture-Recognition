from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "models" / "hand_landmarker.task"
URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists() and DESTINATION.stat().st_size > 1_000_000:
        print(f"Already present: {DESTINATION}")
        return
    with tempfile.NamedTemporaryFile(delete=False, suffix=".task") as temporary:
        temporary_path = Path(temporary.name)
    try:
        urllib.request.urlretrieve(URL, temporary_path)
        if temporary_path.stat().st_size < 1_000_000:
            raise RuntimeError("The downloaded MediaPipe model is unexpectedly small.")
        digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
        temporary_path.replace(DESTINATION)
        print(f"Downloaded {DESTINATION.name} ({DESTINATION.stat().st_size} bytes)")
        print(f"SHA256: {digest}")
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
