from __future__ import annotations

import json

import numpy as np

from backend.artifact_integrity import ArtifactRegistry
from backend.config import MODELS_DIRECTORY, load_runtime_config
from backend.model_runtime import ModelManager


def test_artifact_manifest_and_onnx_metadata_are_current():
    verification = ArtifactRegistry().verify()
    assert verification["verified"], verification["errors"]
    metadata = json.loads(
        (MODELS_DIRECTORY / "gesture_mlp_onnx_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["format"] == "ONNX"
    assert metadata["parity"]["passed"] is True
    assert metadata["parity"]["prediction_agreement"] >= 0.99


def test_model_manager_loads_only_qualified_onnx_runtime():
    config = load_runtime_config()
    manager = ModelManager(config)
    assert manager.ready, manager.errors
    assert manager.status()["available_models"] == ["ONNX"]
    assert manager.status()["selectable_models"] == ["ONNX"]
    probabilities, model_name, _ = manager.predict(
        np.zeros(len(config.feature_names), dtype=np.float32)
    )
    assert model_name == "ONNX"
    assert probabilities.shape == (len(config.class_names),)
    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(), 1.0)


def test_handover_folder_contains_no_python_pickle_model():
    assert not list(MODELS_DIRECTORY.glob("*.joblib"))
    assert not list(MODELS_DIRECTORY.glob("*.pkl"))
