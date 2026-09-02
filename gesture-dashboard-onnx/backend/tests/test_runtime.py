import numpy as np

from backend.config import RuntimeConfig
from backend.model_runtime import InferenceEngine
from backend.runtime import TemporalGate, probability_ema


def test_probability_ema_keeps_rows_normalized():
    rows = probability_ema(np.asarray([[0.9, 0.1], [0.7, 0.3]]), alpha=0.65)
    assert rows.shape == (2, 2)
    assert np.allclose(rows.sum(axis=1), 1.0)
    assert rows[-1, 0] > rows[-1, 1]


def test_temporal_gate_requires_three_stable_frames():
    config = RuntimeConfig()
    gate = TemporalGate(config)
    row = np.zeros(len(config.class_names), dtype=np.float64)
    row[config.class_to_idx["call"]] = 0.95
    row[config.class_to_idx["no_gesture"]] = 0.05
    first = gate.update(row, now=1.0)
    second = gate.update(row, now=1.1)
    third = gate.update(row, now=1.2)
    assert first.execute is False
    assert second.execute is False
    assert third.execute is True
    assert third.predicted_gesture == "call"


def test_negative_class_never_executes():
    config = RuntimeConfig()
    gate = TemporalGate(config)
    row = np.zeros(len(config.class_names), dtype=np.float64)
    row[config.class_to_idx["no_gesture"]] = 1.0
    decision = None
    for index in range(5):
        decision = gate.update(row, now=float(index))
    assert decision is not None
    assert decision.execute is False
    assert decision.reason == "negative class"


def test_ten_fps_budget_verdict_uses_total_pipeline_time():
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config = RuntimeConfig(target_fps=10.0)
    passing = engine._timing(
        mediapipe_ms=33.9,
        feature_ms=0.2,
        classifier_ms=27.761,
        diagnostics_ms=27.761,
        total_ms=61.66,
        effective_fps=10.0,
    )
    assert passing["ten_fps_capacity_pass"] is True
    assert passing["verdict"] == "PASS"
    assert abs(passing["uncapped_fps"] - 16.218) < 0.01
    assert abs(passing["budget_used_percent"] - 61.66) < 0.01

    failing = engine._timing(
        mediapipe_ms=80.0,
        feature_ms=1.0,
        classifier_ms=30.0,
        diagnostics_ms=30.0,
        total_ms=111.0,
        effective_fps=9.0,
    )
    assert failing["ten_fps_capacity_pass"] is False
    assert failing["verdict"] == "FAIL"
