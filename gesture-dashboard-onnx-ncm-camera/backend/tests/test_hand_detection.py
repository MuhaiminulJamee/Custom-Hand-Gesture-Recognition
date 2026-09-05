import numpy as np
import pytest

from backend.hand_detection import HandDetector, hand_pixel_metrics, restore_points
from backend.model_runtime import RuntimeSession
from backend.config import RuntimeConfig
from backend.tests.test_camera_stability import representative_hand


@pytest.mark.parametrize("turns", range(4))
def test_rotated_crop_returns_original_camera_axes(turns):
    original = np.array([[.2, .3], [.8, .7]], dtype=np.float32)
    rotated = original.copy()
    for _ in range(turns):
        rotated = np.column_stack([rotated[:, 1], 1 - rotated[:, 0]])
    restored = restore_points(rotated, (100, 50, 200, 150, turns, 0), (480, 640, 3))
    np.testing.assert_allclose(restored * [640, 480], original * [200, 150] + [100, 50], atol=1e-4)


def test_hand_pixel_measurement_is_not_crop_magnification():
    local = representative_hand()
    restored = restore_points(local, (100, 100, 80, 80, 0, 0), (640, 640, 3))
    metrics = hand_pixel_metrics(restored, (640, 640, 3))
    expected = hand_pixel_metrics(local, (80, 80, 3))
    assert metrics["palm_scale_px"] == pytest.approx(expected["palm_scale_px"], rel=1e-6)
    assert metrics["hand_span_px"] == pytest.approx(expected["hand_span_px"], rel=1e-6)


def test_distant_complete_landmarks_are_not_rejected_by_old_area_gate():
    points = (representative_hand() - .5) * .16 + .5
    assert np.prod(np.ptp(points, axis=0)) < .018
    accepted, diagnostics = RuntimeSession(RuntimeConfig()).stabilize_landmarks(points, 1.0)
    assert accepted is not None
    assert diagnostics["accepted"]


def test_degenerate_and_clipped_distant_landmarks_still_rejected():
    session = RuntimeSession(RuntimeConfig())
    assert session.stabilize_landmarks(np.full((21, 2), .5), 1.0)[0] is None
    assert session.stabilize_landmarks(representative_hand() + 2, 1.0)[0] is None


def simulated_detector():
    detector = HandDetector.__new__(HandDetector)
    detector.ready = True
    detector._shape = detector._view = None
    detector._search_index = detector._misses = detector._unqualified_frames = 0
    detector._recovery_tracking = detector._pose_recovery_requested = False
    return detector


def test_recovery_can_replace_bad_tracked_fingers_with_new_image_evidence(monkeypatch):
    detector = simulated_detector()
    initial = representative_hand()
    recovered = initial - [.1, 0]
    calls = []
    def run(rgb, view, *, video):
        calls.append(video)
        return (.10, initial if video else recovered, None)
    monkeypatch.setattr(detector, '_run_view', run)
    result = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8),
                             candidate_score=lambda points: float(points[0, 0] < .45))
    np.testing.assert_array_equal(result, recovered)
    assert calls == [True, False]
    assert detector.diagnostics()['recovery_attempted']


def test_recovery_is_skipped_when_first_pass_uses_its_time_allowance(monkeypatch):
    import backend.hand_detection as module
    detector = simulated_detector()
    times = iter([1., 1.056])
    monkeypatch.setattr(module.time, 'perf_counter', lambda: next(times))
    calls = []
    def run(rgb, view, *, video):
        calls.append(video)
        return None
    monkeypatch.setattr(detector, '_run_view', run)
    assert detector.detect(np.zeros((640, 640, 3), dtype=np.uint8)) is None
    assert calls == [True]
