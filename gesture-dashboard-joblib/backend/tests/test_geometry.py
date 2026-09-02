import numpy as np

from backend.geometry import landmarks_to_feature


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
