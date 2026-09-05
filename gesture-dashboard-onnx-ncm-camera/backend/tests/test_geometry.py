import numpy as np
import pytest

import backend.geometry as geometry
from backend.config import RuntimeConfig
from backend.geometry import GeometryResolver, landmarks_to_feature
from backend.runtime import TemporalGate


def representative_hand():
    # Non-degenerate synthetic 21-point hand geometry for feature-contract tests.
    points = np.zeros((21, 2), dtype=np.float32)
    points[0] = [0.5, 0.9]
    bases = [0.32, 0.43, 0.54, 0.65, 0.76]
    for finger, base_x in enumerate(bases):
        start = 1 if finger == 0 else 5 + (finger - 1) * 4
        count = 4
        for joint in range(count):
            points[start + joint] = [base_x + 0.015 * joint, 0.78 - 0.13 * joint]
    return points


def test_feature_contract_is_76_dimensions():
    feature = landmarks_to_feature(representative_hand())
    assert feature.shape == (76,)
    assert np.isfinite(feature).all()


def test_features_are_translation_and_scale_invariant_except_screen_directions():
    points = representative_hand()
    translated_scaled = points * 1.7 + np.asarray([0.2, -0.1], dtype=np.float32)
    first = landmarks_to_feature(points)
    second = landmarks_to_feature(translated_scaled)
    assert np.allclose(first, second, atol=1e-5)


def test_ncm_horizontal_semantics_are_swapped_without_mirroring(monkeypatch):
    config = RuntimeConfig()
    resolver = GeometryResolver(config)
    probabilities = np.zeros(len(config.class_names), dtype=np.float64)
    probabilities[config.class_to_idx["left"]] = 1.0
    monkeypatch.setattr(geometry, "single_index_pose_score", lambda _points: 1.0)
    monkeypatch.setattr(geometry, "finger_extension_score", lambda *_args: 0.0)

    points = np.zeros((21, 2), dtype=np.float32)
    points[5] = [0.4, 0.5]
    points[8] = [0.8, 0.5]
    positive, positive_details = resolver._directional(probabilities, points)
    points[8] = [0.1, 0.5]
    negative, negative_details = resolver._directional(probabilities, points)

    assert config.class_names[int(np.argmax(positive))] == "left"
    assert config.class_names[int(np.argmax(negative))] == "right"
    assert positive_details["horizontal_mirror"] is False
    assert negative_details["horizontal_mirror"] is False
    assert positive_details["horizontal_semantic_swap"] is True


def side_view_pointing_hand():
    # Foreshortened folded fingers appear straight in 2-D; their tips still
    # end well behind the extended index. This defeated the angle-only gate.
    points = np.zeros((21, 2), dtype=np.float32)
    points[1:5] = [[0.2, -0.2], [0.4, -0.3], [0.6, -0.35], [0.7, -0.3]]
    for mcp, y in ((5, -0.3), (9, 0.0), (13, 0.2), (17, 0.4)):
        step = 0.4 if mcp == 5 else 0.05
        for joint in range(4):
            points[mcp + joint] = [1.0 + step * joint, y]
    return points


@pytest.mark.parametrize("gesture,rotation", [
    ("left", 0), ("right", np.pi),
])
def test_side_view_pointing_reaches_confirmed_action(gesture, rotation):
    config = RuntimeConfig()
    points = side_view_pointing_hand()
    transform = np.asarray([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)],
    ])
    points = (points @ transform.T) * 0.15 + 0.5
    probabilities = np.full(8, 0.01)
    probabilities[config.class_to_idx[gesture]] = 0.93
    resolved, details = GeometryResolver(config).resolve(probabilities, points)
    assert details["directional"]["mean_non_index_extension"] > 0.70
    assert details["directional"]["model_supported_pose"] is True
    assert details["pose_validation"]["valid"] is True
    gate = TemporalGate(config)
    for now in (1.0, 1.1, 1.2):
        decision = gate.update(resolved, now=now, pose_valid=details["pose_validation"]["valid"])
    assert decision.execute is True
    assert decision.predicted_gesture == gesture
    # The new geometry path must never bypass unknown-gesture rejection.
    assert gate.update(resolved, known_gesture_mass=0.1).execute is False


@pytest.mark.parametrize("pose", ["open_palm", "rock", "fist", "diagonal"])
def test_side_view_fallback_rejects_non_pointing_poses(pose):
    config = RuntimeConfig()
    points = side_view_pointing_hand()
    if pose == "open_palm":
        for mcp in (9, 13, 17):
            points[mcp:mcp + 4, 0] = points[5:9, 0]
    elif pose == "rock":
        points[17:21, 0] = points[5:9, 0]
    elif pose == "fist":
        points[5:9, 0] = points[9:13, 0]
    else:
        angle = np.pi / 4
        points = points @ np.asarray([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]]).T
    probabilities = np.zeros(8)
    probabilities[0] = 1.0
    _, details = GeometryResolver(config).resolve(probabilities, points)
    assert details["pose_validation"]["valid"] is False


@pytest.mark.parametrize("probabilities", [
    [0.6, 0.4, 0, 0, 0, 0, 0, 0],
    [0.01, 0.99, 0, 0, 0, 0, 0, 0],
])
def test_side_view_fallback_requires_confident_matching_model(probabilities):
    _, details = GeometryResolver(RuntimeConfig()).resolve(
        np.asarray(probabilities), side_view_pointing_hand()
    )
    assert details["pose_validation"]["valid"] is False


def test_clear_dorsal_survives_classifier_domain_miss_but_requires_longer_hold():
    config = RuntimeConfig()
    points = 1.0 - representative_hand()
    points[:, 0] *= 0.4  # Four fingers held together, as in the camera regression.
    probabilities = np.eye(8)[config.class_to_idx['like']]
    resolved, details = GeometryResolver(config).resolve(probabilities, points)
    pose = details['pose_validation']
    assert pose['gesture'] == 'dorsal'
    assert pose['valid'] and pose['geometry_supported']
    gate = TemporalGate(config)
    for now in (1.0, 1.1, 1.2, 1.3):
        decision = gate.update(resolved, now=now, known_gesture_mass=.01,
                               pose_valid=pose['valid'], geometry_supported=pose['geometry_supported'])
        assert not decision.execute
    decision = gate.update(resolved, now=1.4, known_gesture_mass=.01,
                           pose_valid=True, geometry_supported=True)
    assert decision.execute
    assert decision.predicted_gesture == 'dorsal'
    assert decision.known_gesture_mass == .01  # Never falsify the classifier score.
    assert decision.reason == 'stable Dorsal geometry'
    assert not gate.update(resolved, known_gesture_mass=.01, geometry_supported=False).execute


def test_geometry_support_never_bypasses_invalid_pose_or_other_class_rejection():
    gate = TemporalGate(RuntimeConfig())
    assert not gate.update(np.eye(8)[0], known_gesture_mass=.01, geometry_supported=True).execute
    decision = gate.update(np.eye(8)[6], known_gesture_mass=.01, geometry_supported=True, pose_valid=False)
    assert decision.predicted_gesture == 'no_gesture'
