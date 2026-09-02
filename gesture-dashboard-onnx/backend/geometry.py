from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RuntimeConfig


def _unit_direction(vector: np.ndarray, description: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError(f"Degenerate {description} direction.")
    return np.asarray(vector, dtype=np.float32) / norm


def thumb_index_gap_ratio(landmarks_xy: np.ndarray) -> float:
    points = np.asarray(landmarks_xy, dtype=np.float32).reshape(21, 2)
    references = np.asarray([
        np.linalg.norm(points[5] - points[0]),
        np.linalg.norm(points[9] - points[0]),
        np.linalg.norm(points[17] - points[0]),
        np.linalg.norm(points[17] - points[5]),
    ], dtype=np.float32)
    positive = references[references > 1e-8]
    if not len(positive):
        raise ValueError("Degenerate palm scale for thumb-index gap.")
    return float(np.linalg.norm(points[4] - points[8]) / np.median(positive))


def joint_angle_degrees(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    first = np.asarray(point_a, dtype=np.float32) - np.asarray(point_b, dtype=np.float32)
    second = np.asarray(point_c, dtype=np.float32) - np.asarray(point_b, dtype=np.float32)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def finger_extension_score(points: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> float:
    mean_angle = 0.5 * (
        joint_angle_degrees(points[mcp], points[pip], points[dip])
        + joint_angle_degrees(points[pip], points[dip], points[tip])
    )
    return float(np.clip((mean_angle - 110.0) / 60.0, 0.0, 1.0))


def landmarks_to_feature(landmarks_xy: np.ndarray) -> np.ndarray:
    """Exact 76-D geometry used by the supplied v18 training notebook."""
    points = np.asarray(landmarks_xy, dtype=np.float32).reshape(21, 2).copy()
    raw_points = points.copy()
    pinch_gap_ratio = thumb_index_gap_ratio(points)

    thumb_extension = finger_extension_score(raw_points, 1, 2, 3, 4)
    finger_extensions = np.asarray([
        finger_extension_score(raw_points, 5, 6, 7, 8),
        finger_extension_score(raw_points, 9, 10, 11, 12),
        finger_extension_score(raw_points, 13, 14, 15, 16),
        finger_extension_score(raw_points, 17, 18, 19, 20),
    ], dtype=np.float32)
    other_fingers_folded = float(1.0 - finger_extensions[1:].mean())
    palm_references = np.asarray([
        np.linalg.norm(raw_points[5] - raw_points[0]),
        np.linalg.norm(raw_points[9] - raw_points[0]),
        np.linalg.norm(raw_points[17] - raw_points[0]),
        np.linalg.norm(raw_points[17] - raw_points[5]),
    ], dtype=np.float32)
    positive_references = palm_references[palm_references > 1e-8]
    if not len(positive_references):
        raise ValueError("Degenerate palm scale for engineered hand shape.")
    palm_scale = float(np.median(positive_references))
    shape_features = np.asarray([
        thumb_extension,
        *finger_extensions,
        other_fingers_folded,
        float(np.linalg.norm(raw_points[4] - raw_points[5]) / palm_scale),
        float(np.linalg.norm(raw_points[4] - raw_points[0]) / palm_scale),
    ], dtype=np.float32)

    points -= points[0]
    orientation_features = np.concatenate([
        _unit_direction(points[8] - points[5], "index MCP-to-tip"),
        _unit_direction(points[8], "wrist-to-index-tip"),
    ])
    middle_mcp = points[9]
    if np.linalg.norm(middle_mcp) < 1e-8:
        raise ValueError("Degenerate wrist-to-middle-finger geometry.")
    angle = -np.pi / 2 - np.arctan2(middle_mcp[1], middle_mcp[0])
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    points = points @ rotation.T
    scale = float(np.linalg.norm(points, axis=1).max())
    if scale < 1e-8:
        raise ValueError("Degenerate landmark scale.")
    points /= scale
    if points[5, 0] < points[17, 0]:
        points[:, 0] *= -1
    radial_distances = np.linalg.norm(points, axis=1)
    feature = np.concatenate([
        points.reshape(-1),
        radial_distances,
        orientation_features,
        np.asarray([pinch_gap_ratio], dtype=np.float32),
        shape_features,
    ]).astype(np.float32)
    if feature.shape != (76,):
        raise AssertionError(f"Expected 76 features, produced {feature.shape}.")
    return feature


def single_index_pose_score(landmarks_xy: np.ndarray) -> float:
    points = np.asarray(landmarks_xy, dtype=np.float32).reshape(21, 2)
    index_extension = finger_extension_score(points, 5, 6, 7, 8)
    other_extensions = np.asarray([
        finger_extension_score(points, 9, 10, 11, 12),
        finger_extension_score(points, 13, 14, 15, 16),
        finger_extension_score(points, 17, 18, 19, 20),
    ])
    other_fingers_folded = float(1.0 - other_extensions.mean())
    palm_scale = max(float(np.linalg.norm(points[9] - points[0])), 1e-6)
    index_reach = float(np.linalg.norm(points[8] - points[0]))
    other_tip_reach = float(max(
        np.linalg.norm(points[12] - points[0]),
        np.linalg.norm(points[16] - points[0]),
        np.linalg.norm(points[20] - points[0]),
    ))
    prominence = float(np.clip(
        ((index_reach - other_tip_reach) / palm_scale + 0.10) / 0.70,
        0.0, 1.0,
    ))
    return float(np.clip(
        0.50 * index_extension + 0.35 * other_fingers_folded + 0.15 * prominence,
        0.0, 1.0,
    ))


def return_main_pose_geometry(landmarks_xy: np.ndarray) -> dict[str, float | int | bool]:
    points = np.asarray(landmarks_xy, dtype=np.float32).reshape(21, 2)
    fingers = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    extensions = np.asarray([
        finger_extension_score(points, *finger) for finger in fingers
    ], dtype=np.float32)
    vectors = np.asarray([points[tip] - points[mcp] for mcp, _, _, tip in fingers])
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms < 1e-8):
        return {"score": 0.0, "strong_geometry": False, "downward_finger_count": 0}
    units = vectors / norms[:, None]
    downward_count = int(((units[:, 1] >= 0.50) & (extensions >= 0.50)).sum())
    mean_direction = units.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_direction))
    if mean_norm < 1e-8:
        downward_score = parallel_score = 0.0
    else:
        mean_direction /= mean_norm
        downward_score = float(np.clip((mean_direction[1] - 0.30) / 0.60, 0.0, 1.0))
        parallel_score = float(np.clip((mean_norm - 0.70) / 0.30, 0.0, 1.0))
    palm_scale = max(float(np.linalg.norm(points[9] - points[0])), 1e-6)
    tips = (8, 12, 16, 20)
    mean_gap = float(np.mean([
        np.linalg.norm(points[right] - points[left]) / palm_scale
        for left, right in zip(tips[:-1], tips[1:])
    ]))
    together = float(np.clip((0.78 - mean_gap) / 0.52, 0.0, 1.0))
    extended = float(extensions.mean())
    minimum = float(extensions.min())
    base = (
        0.48 * (0.70 * extended + 0.30 * minimum)
        + 0.27 * downward_score
        + 0.15 * together
        + 0.10 * parallel_score
    )
    score = float(np.clip(base * (0.75 + 0.25 * together), 0.0, 1.0))
    strong = bool(
        score >= 0.76 and downward_count >= 4 and minimum >= 0.50
        and downward_score >= 0.72 and together >= 0.18
    )
    return {
        "score": score,
        "strong_geometry": strong,
        "downward_finger_count": downward_count,
        "extended_score": extended,
        "minimum_extension_score": minimum,
        "downward_score": downward_score,
        "together_score": together,
        "parallel_score": parallel_score,
    }


def zoom_pose_geometry(landmarks_xy: np.ndarray) -> dict[str, float | bool]:
    points = np.asarray(landmarks_xy, dtype=np.float32).reshape(21, 2)
    thumb_extension = finger_extension_score(points, 1, 2, 3, 4)
    index_extension = finger_extension_score(points, 5, 6, 7, 8)
    others = np.asarray([
        finger_extension_score(points, 9, 10, 11, 12),
        finger_extension_score(points, 13, 14, 15, 16),
        finger_extension_score(points, 17, 18, 19, 20),
    ])
    folded = float(1.0 - others.mean())
    references = np.asarray([
        np.linalg.norm(points[5] - points[0]), np.linalg.norm(points[9] - points[0]),
        np.linalg.norm(points[17] - points[0]), np.linalg.norm(points[17] - points[5]),
    ])
    positive = references[references > 1e-8]
    if not len(positive):
        return {"score": 0.0, "strong_geometry": False, "thumb_opposition_score": 0.0, "pair_reach_score": 0.0}
    scale = float(np.median(positive))
    thumb_reach = float(np.linalg.norm(points[4] - points[0]) / scale)
    index_reach = float(np.linalg.norm(points[8] - points[0]) / scale)
    thumb_mcp_ratio = float(np.linalg.norm(points[4] - points[5]) / scale)
    other_reaches = np.asarray([
        np.linalg.norm(points[12] - points[0]) / scale,
        np.linalg.norm(points[16] - points[0]) / scale,
        np.linalg.norm(points[20] - points[0]) / scale,
    ])
    thumb_reach_score = float(np.clip((thumb_reach - 0.65) / 0.75, 0.0, 1.0))
    index_reach_score = float(np.clip((index_reach - 0.80) / 0.90, 0.0, 1.0))
    pair_reach = float(np.clip((min(thumb_reach, index_reach) - 0.75) / 0.70, 0.0, 1.0))
    prominence = 0.5 * (thumb_reach + index_reach) - float(other_reaches.mean())
    prominence_score = float(np.clip((prominence + 0.05) / 0.55, 0.0, 1.0))
    opposition = float(np.clip((thumb_mcp_ratio - 0.38) / 0.72, 0.0, 1.0))
    score = float(np.clip(
        0.27 * folded + 0.10 * thumb_extension + 0.11 * index_extension
        + 0.12 * thumb_reach_score + 0.11 * index_reach_score
        + 0.10 * pair_reach + 0.08 * prominence_score + 0.11 * opposition,
        0.0, 1.0,
    ))
    strong = bool(
        score >= 0.60 and folded >= 0.35 and thumb_reach_score >= 0.25
        and index_reach_score >= 0.20 and pair_reach >= 0.15
        and prominence_score >= 0.08 and opposition >= 0.25
    )
    return {
        "score": score,
        "strong_geometry": strong,
        "thumb_extension": thumb_extension,
        "index_extension": index_extension,
        "other_fingers_folded": folded,
        "thumb_reach_score": thumb_reach_score,
        "index_reach_score": index_reach_score,
        "pair_reach_score": pair_reach,
        "pair_prominence_score": prominence_score,
        "thumb_opposition_score": opposition,
        "thumb_tip_index_mcp_ratio": thumb_mcp_ratio,
    }


def _set_probability_floor(probabilities: np.ndarray, target_index: int, floor: float) -> np.ndarray:
    adjusted = np.asarray(probabilities, dtype=np.float64).copy()
    if adjusted[target_index] < floor:
        other_total = float(adjusted.sum() - adjusted[target_index])
        if other_total > 1e-12:
            adjusted *= (1.0 - floor) / other_total
        else:
            adjusted.fill((1.0 - floor) / max(1, len(adjusted) - 1))
        adjusted[target_index] = floor
    adjusted /= max(float(adjusted.sum()), 1e-12)
    return adjusted


@dataclass(slots=True)
class GeometryResolver:
    config: RuntimeConfig

    def _directional(self, probabilities: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, dict | None]:
        points = np.asarray(landmarks, dtype=np.float32).reshape(21, 2)
        geometry = zoom_pose_geometry(points)
        raw = self.config.class_names[int(np.argmax(probabilities))]
        strong_zoom = bool(
            geometry["score"] >= 0.48 and geometry["thumb_opposition_score"] >= 0.18
            and (raw in {"zoom_in", "zoom_out"} or (
                geometry["strong_geometry"]
                and geometry["thumb_opposition_score"] >= 0.45
                and geometry["pair_reach_score"] >= 0.25
            ))
        )
        if strong_zoom:
            return probabilities, None
        pose_score = single_index_pose_score(points)
        vector = points[8] - points[5]
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            return probabilities, None
        non_index_extensions = np.asarray([
            finger_extension_score(points, 9, 10, 11, 12),
            finger_extension_score(points, 13, 14, 15, 16),
            finger_extension_score(points, 17, 18, 19, 20),
        ], dtype=np.float32)
        middle_extension = float(non_index_extensions[0])
        index_reach = max(float(np.linalg.norm(points[8] - points[0])), 1e-8)
        peace = middle_extension >= 0.60 and float(np.linalg.norm(points[12] - points[0]) / index_reach) >= 0.72
        dominance = float(np.max(np.abs(vector)) / norm)
        if (
            pose_score < 0.72
            or dominance < 0.78
            or peace
            or float(non_index_extensions.max()) > 0.55
        ):
            return probabilities, None
        dx, dy = map(float, vector)
        gesture = ("one_down" if dy > 0 else "one") if abs(dy) >= abs(dx) else ("one_right" if dx > 0 else "one_left")
        floor = min(0.98, 0.92 + 0.06 * max(0.0, pose_score - 0.72) / 0.28)
        return _set_probability_floor(probabilities, self.config.class_to_idx[gesture], floor), {
            "gesture": gesture, "pose_score": pose_score, "axis_dominance": dominance,
        }

    def _zoom(self, probabilities: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, dict | None]:
        calibration = self.config.zoom_gap_calibration
        if not calibration:
            return probabilities, None
        adjusted = np.asarray(probabilities, dtype=np.float64).copy()
        indexes = self.config.class_to_idx
        raw = self.config.class_names[int(np.argmax(adjusted))]
        geometry = zoom_pose_geometry(landmarks)
        combined = float(adjusted[indexes["zoom_in"]] + adjusted[indexes["zoom_out"]])
        credible = bool(geometry["score"] >= 0.48 and geometry["thumb_opposition_score"] >= 0.18)
        if raw in {"zoom_in", "zoom_out"} and not credible:
            adjusted[indexes["zoom_in"]] = adjusted[indexes["zoom_out"]] = 0.0
            total = float(adjusted.sum())
            if total <= 1e-12:
                adjusted[indexes["no_gesture"]] = 1.0
            else:
                adjusted /= total
            return adjusted, None
        rescue = bool(
            (raw == "no_gesture" and combined >= 0.10 and geometry["strong_geometry"])
            or (
                raw in {"one", "one_down", "one_left", "one_right"}
                and combined >= 0.005 and geometry["strong_geometry"]
                and geometry["thumb_opposition_score"] >= 0.45
                and geometry["pair_reach_score"] >= 0.25
            )
        )
        if raw not in {"zoom_in", "zoom_out"} and not rescue:
            return adjusted, None
        gap = thumb_index_gap_ratio(landmarks)
        threshold = float(calibration["threshold"])
        gesture = "zoom_in" if gap >= threshold else "zoom_out"
        margin = float(calibration.get("hysteresis_margin", 0.025))
        floor = min(0.98, 0.90 + 0.06 * min(1.0, abs(gap - threshold) / max(margin, 1e-6)))
        adjusted = _set_probability_floor(adjusted, indexes[gesture], floor)
        return adjusted, {
            "gesture": gesture, "gap_ratio": gap, "threshold": threshold,
            "pose_score": float(geometry["score"]),
            "thumb_opposition_score": float(geometry["thumb_opposition_score"]),
            "combined_zoom_probability": combined,
        }

    def _return_main(self, probabilities: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, dict | None]:
        adjusted = np.asarray(probabilities, dtype=np.float64).copy()
        geometry = return_main_pose_geometry(landmarks)
        target = self.config.class_to_idx["dorsal_hand"]
        raw = self.config.class_names[int(np.argmax(adjusted))]
        supported = bool(
            geometry["score"] >= 0.70 and geometry["downward_finger_count"] >= 4
            and adjusted[target] >= 0.05
        )
        if raw not in {"dorsal_hand", "palm", "no_gesture"} or not (geometry["strong_geometry"] or supported):
            return adjusted, None
        floor = min(0.98, 0.92 + 0.05 * max(0.0, float(geometry["score"]) - 0.70) / 0.30)
        return _set_probability_floor(adjusted, target, floor), geometry

    def resolve(self, probabilities: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, dict[str, dict | None]]:
        adjusted, directional = self._directional(probabilities, landmarks)
        adjusted, zoom = self._zoom(adjusted, landmarks)
        adjusted, return_main = self._return_main(adjusted, landmarks)
        return adjusted, {
            "directional": directional,
            "zoom": zoom,
            "return_main": return_main,
        }
