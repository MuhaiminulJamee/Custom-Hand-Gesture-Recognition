from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIRECTORY = PROJECT_ROOT / ".runtime"
MATPLOTLIB_DIRECTORY = RUNTIME_DIRECTORY / "matplotlib"
MATPLOTLIB_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_DIRECTORY))
os.environ.setdefault("GLOG_minloglevel", "2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.artifact_integrity import ArtifactRegistry
from backend.config import load_production_settings, load_runtime_config
from backend.model_runtime import InferenceEngine
from backend.release_qualification import ReleaseQualifier


def main() -> int:
    models_directory = PROJECT_ROOT / "models"
    registry = ArtifactRegistry(models_directory)
    integrity = registry.verify()
    if not integrity["verified"]:
        print("Artifact integrity: FAIL")
        for error in integrity["errors"]:
            print(f"  - {error}")
        return 1
    print(
        "Artifact integrity: PASS | release fingerprint: "
        + str(integrity["release_fingerprint"])
    )
    try:
        config = load_runtime_config(models_directory)
        engine = InferenceEngine(config, models_directory)
        status = engine.status()
        if not status["ready"]:
            for error in status["model"]["errors"]:
                print(f"  - {error}")
            if status["mediapipe"]["error"]:
                print(f"  - {status['mediapipe']['error']}")
            return 1
        qualification = ReleaseQualifier(
            config, load_production_settings(), models_directory
        ).status(engine.model_manager.selected_model_name)
        if not qualification.get("current") or not qualification.get("pc_release_ready"):
            print(
                "  - Frozen production qualification is missing, stale, or failed. "
                "Run scripts\\qualify_release.py --require-pass, then rebuild the manifest."
            )
            return 1
        print(
            "Frozen quality gate: PASS | model: "
            + str(qualification.get("selected_model"))
            + f" | gated macro-F1: {float(qualification['gated_macro_f1']):.4f}"
        )
        replay_path = models_directory / "gesture_online_replay_cache.npz"
        if replay_path.exists():
            with np.load(replay_path, allow_pickle=False) as cache:
                sample = cache["X"][:1].astype(np.float32)
            probabilities, name, _ = engine.model_manager.predict(sample)
            if probabilities.shape != (len(config.class_names),):
                raise ValueError("Classifier output does not match the 15-class contract.")
            if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0):
                raise ValueError("Classifier smoke-test probabilities are invalid.")
            print(f"Classifier smoke test: PASS | model: {name}")
        print("MediaPipe initialization: PASS")
        engine.close()
        return 0
    except Exception as error:
        print(f"Production preflight: FAIL | {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
