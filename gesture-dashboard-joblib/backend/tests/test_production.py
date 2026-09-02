from pathlib import Path

import pytest

from backend.artifact_integrity import ArtifactRegistry
from backend.config import RuntimeConfig, load_production_settings
from backend.model_runtime import ModelManager
from backend.observability import RuntimeMetrics


def test_artifact_manifest_detects_runtime_mutation(tmp_path: Path):
    (tmp_path / "hand_landmarker.task").write_bytes(b"hand-model")
    (tmp_path / "gesture_joblib_runtime_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "gesture_primary_model.joblib").write_bytes(b"trusted-model")
    (tmp_path / "gesture_production_selection.json").write_text("{}", encoding="utf-8")
    (tmp_path / "gesture_production_qualification.json").write_text("{}", encoding="utf-8")
    registry = ArtifactRegistry(tmp_path)
    manifest = registry.build()
    assert len(manifest["release_fingerprint"]) == 64
    assert registry.verify()["verified"] is True

    (tmp_path / "gesture_primary_model.joblib").write_bytes(b"changed-model")
    failed = registry.verify()
    assert failed["verified"] is False
    assert any("hash changed" in error for error in failed["errors"])


def test_runtime_metrics_exposes_latency_quantiles_and_budget_rate():
    metrics = RuntimeMetrics(window_size=50)
    metrics.session_opened()
    for total_ms in (50.0, 60.0, 70.0, 80.0, 120.0):
        metrics.record_frame({
            "status": "predicted",
            "model": "OnlineMLP",
            "runtime_action": "Wait / No Action",
            "timing": {
                "total_ms": total_ms,
                "mediapipe_ms": total_ms / 2,
                "classifier_ms": total_ms / 3,
                "feature_ms": 0.1,
                "effective_application_fps": 10.0,
                "ten_fps_capacity_pass": total_ms <= 100.0,
            },
        })
    metrics.session_closed()
    snapshot = metrics.snapshot()
    assert snapshot["frames_total"] == 5
    assert snapshot["ten_fps_budget_pass_rate"] == pytest.approx(0.8)
    assert snapshot["timing"]["total_ms"]["p50"] == pytest.approx(70.0)
    assert snapshot["timing"]["total_ms"]["p99"] > 100.0
    assert snapshot["active_sessions"] == 0


def test_production_defaults_disable_force_and_hot_reload(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GESTURE_ENVIRONMENT", "production")
    monkeypatch.delenv("GESTURE_ALLOW_FORCE_LEARNING", raising=False)
    monkeypatch.delenv("GESTURE_ALLOW_ARTIFACT_RELOAD", raising=False)
    settings = load_production_settings()
    assert settings.allow_force_learning is False
    assert settings.allow_artifact_reload is False
    assert settings.require_artifact_manifest is True
    assert settings.deferred_quality_classes == ("no_gesture",)


def test_model_manager_rejects_unknown_explicit_model(tmp_path: Path):
    manager = ModelManager(RuntimeConfig(), tmp_path, load_models=False)
    manager.models["MLP"] = object()
    manager.selected_model_name = "MLP"
    with pytest.raises(ValueError, match="Unknown model selection"):
        manager.resolve_name("not-a-model")


def test_online_update_is_staged_when_online_model_is_the_qualified_runtime(tmp_path: Path):
    manager = ModelManager(RuntimeConfig(), tmp_path, load_models=False)
    qualified = object()
    candidate = object()
    manager.models["OnlineMLP"] = qualified
    manager.selected_model_name = "OnlineMLP"
    manager.activate_online_model(candidate, select=False)
    assert manager.models["OnlineMLP"] is qualified
    assert manager.models["OnlineMLPCandidate"] is candidate
    assert "OnlineMLPCandidate" not in manager.status()["selectable_models"]
