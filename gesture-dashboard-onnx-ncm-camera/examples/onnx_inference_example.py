from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import load_runtime_config
from backend.model_runtime import ModelManager


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the qualified gesture ONNX model on one 76-D feature vector."
    )
    parser.add_argument(
        "--features-json",
        required=True,
        type=Path,
        help="JSON file containing exactly 76 finite numbers.",
    )
    arguments = parser.parse_args()
    feature = np.asarray(
        json.loads(arguments.features_json.read_text(encoding="utf-8")),
        dtype=np.float32,
    )
    if feature.shape != (76,) or not np.isfinite(feature).all():
        raise SystemExit("The feature JSON must contain exactly 76 finite numbers.")

    config = load_runtime_config()
    manager = ModelManager(config)
    if not manager.ready:
        raise SystemExit("; ".join(manager.errors))
    probabilities, runtime_name, elapsed_ms = manager.predict(feature)
    best_index = int(np.argmax(probabilities))
    print(f"Runtime: {runtime_name}")
    print(f"Prediction: {config.class_names[best_index]}")
    print(f"Confidence: {probabilities[best_index]:.6f}")
    print(f"Classifier time: {elapsed_ms:.3f} ms")
    print(json.dumps({
        name: float(probabilities[index])
        for index, name in enumerate(config.class_names)
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
