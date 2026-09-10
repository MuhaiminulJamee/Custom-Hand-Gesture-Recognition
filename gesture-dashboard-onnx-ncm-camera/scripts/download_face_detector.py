"""Install the pinned MediaPipe face guard model from its official source."""
import hashlib
from pathlib import Path
import urllib.request

URL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
SHA256 = 'b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f'


def main():
    path = Path(__file__).resolve().parents[1] / 'models/blaze_face_short_range.tflite'
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == SHA256:
        print('Face guard model already verified.')
        return
    with urllib.request.urlopen(URL, timeout=30) as response:
        data = response.read(1_000_000)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError('Face detector download checksum mismatch.')
    temporary = path.with_suffix('.download')
    temporary.write_bytes(data)
    temporary.replace(path)
    print(f'Installed {path.name} ({len(data)} bytes).')


if __name__ == '__main__':
    main()
