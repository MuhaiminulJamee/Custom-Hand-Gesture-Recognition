from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MATPLOTLIB_CACHE = PROJECT_ROOT / ".runtime" / "matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

from backend.artifact_integrity import ArtifactRegistry
from backend.config import load_runtime_config
from backend.model_runtime import InferenceEngine


def main() -> int:
    registry = ArtifactRegistry()
    integrity = registry.verify()
    if not integrity["verified"]:
        print("ONNX artifact integrity: FAIL")
        for error in integrity["errors"]:
            print(f"  - {error}")
        return 1
    print(
        "ONNX artifact integrity: PASS | release fingerprint: "
        + str(integrity["release_fingerprint"])
    )

    config = load_runtime_config()
    engine = InferenceEngine(config)
    try:
        status = engine.status()
        if not status["ready"]:
            for error in status["model"]["errors"]:
                print(f"  - {error}")
            if status["mediapipe"].get("error"):
                print("  - " + str(status["mediapipe"]["error"]))
            return 1
        probabilities, name, _ = engine.model_manager.predict(
            np.zeros(len(config.feature_names), dtype=np.float32)
        )
        if name != "ONNX" or probabilities.shape != (len(config.class_names),):
            print("ONNX classifier smoke test returned an unexpected contract.")
            return 1
        if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0):
            print("ONNX classifier smoke test returned invalid probabilities.")
            return 1
        print("ONNX graph/session load: PASS | provider: CPUExecutionProvider")
        print("MediaPipe initialization: PASS")
        print("Runtime contract: PASS | 76 float32 features -> 15 probabilities")
        return 0
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
