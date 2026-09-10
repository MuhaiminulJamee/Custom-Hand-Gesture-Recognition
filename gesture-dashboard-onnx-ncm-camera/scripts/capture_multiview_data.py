"""Interactively capture reviewed multi-angle directional gestures.

Examples (run from the project root):

    python -m scripts.capture_multiview_data --participant operator-001 --source ncm
    python -m scripts.capture_multiview_data --participant operator-002 --source webcam

The operator, not the current model prediction, supplies the ground-truth label.
Each accepted frame keeps the original JPEG plus exact landmarks/features in a
CSV manifest.  Press Space to accept, N to skip the current label/view bucket,
or Esc to stop cleanly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import load_runtime_config
from backend.geometry import landmarks_to_feature, single_index_pose_score
from backend.hand_detection import HandDetector
from backend.model_runtime import adaptive_low_light_preprocess
from research.multiview import (
    CAPTURE_SPLITS,
    DIRECTION_LABELS,
    MANIFEST_FIELDS,
    VIEWPOINTS,
    direction_from_landmarks,
    participant_split,
)


VIEW_INSTRUCTIONS = {
    "front": "Palm plane approximately facing the camera",
    "yaw_left": "Turn the hand about 25 degrees to the operator's left",
    "yaw_right": "Turn the hand about 25 degrees to the operator's right",
    "pitch_toward": "Tilt the fingertips slightly toward the camera",
    "pitch_away": "Tilt the fingertips slightly away from the camera",
    "roll_left": "Relax the wrist about 15 degrees counter-clockwise",
    "roll_right": "Relax the wrist about 15 degrees clockwise",
    "casual": "Use a natural imperfect pose; do not align it precisely",
}


class WebcamSource:
    def __init__(self, device: int):
        import cv2

        self.capture = cv2.VideoCapture(device)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open webcam device {device}")

    def read(self) -> tuple[bytes, str]:
        import cv2

        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("Webcam stopped returning frames")
        encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not encoded_ok:
            raise RuntimeError("Could not encode webcam frame")
        return encoded.tobytes(), str(time.monotonic_ns())

    def close(self) -> None:
        self.capture.release()


class NcmHttpSource:
    def __init__(self, url: str):
        self.url = url

    def read(self) -> tuple[bytes, str]:
        with urllib.request.urlopen(self.url, timeout=3) as response:
            return response.read(), response.headers.get("X-NCM-Frame-ID", "")

    def close(self) -> None:
        return None


def _safe_identifier(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in normalized):
        raise ValueError(f"{name} may contain only letters, numbers, hyphens, and underscores")
    return normalized


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("Existing manifest columns do not match the multi-view schema")
        return list(reader)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _existing_split(rows: list[dict[str, str]], participant: str) -> str | None:
    splits = {row["split"] for row in rows if row["participant_id"] == participant}
    if len(splits) > 1:
        raise ValueError(f"Existing manifest already leaks participant {participant!r} across splits")
    return next(iter(splits), None)


def _prepare_frame(
    jpeg: bytes,
    detector: HandDetector,
    crop_ratio: float,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    import cv2

    source = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError("Camera frame is not a valid JPEG")
    height, width = source.shape[:2]
    side = max(1, round(min(height, width) * crop_ratio))
    left = max(0, (width - side) // 2)
    top = max(0, (height - side) // 2)
    analysis = source[top : top + side, left : left + side]
    if max(analysis.shape[:2]) > 960:
        scale = 960 / max(analysis.shape[:2])
        analysis = cv2.resize(
            analysis,
            (round(analysis.shape[1] * scale), round(analysis.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    detection_image, quality = adaptive_low_light_preprocess(analysis)
    rgb = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)
    points = detector.detect(rgb)
    diagnostics = detector.diagnostics()
    details: dict[str, Any] = {
        "source_width": width,
        "source_height": height,
        "input_luma_median": float(quality["input_luma_median"]),
        "sharpness": float(quality["sharpness_laplacian_variance"]),
        "quality_issues": [
            name
            for name, active in (
                ("near_blank", quality["near_blank"]),
                ("low_contrast", quality["low_contrast"]),
                ("blur_risk", quality["blur_risk"]),
            )
            if active
        ],
        "handedness": diagnostics.get("selected_handedness") or "unknown",
        "detector_rejection": diagnostics.get("rejection_reason"),
        "analysis_image": analysis,
    }
    if points is None:
        return source, None, details
    details["pose_score"] = float(single_index_pose_score(points))
    direction, dominance = direction_from_landmarks(points)
    details["captured_direction"] = direction
    details["axis_dominance"] = dominance
    details["feature"] = landmarks_to_feature(points)
    return source, points, details


def _draw_preview(
    image: np.ndarray,
    points: np.ndarray | None,
    details: dict[str, Any],
    *,
    label: str,
    view: str,
    captured: int,
    target: int,
) -> np.ndarray:
    import cv2

    preview = image.copy()
    if points is not None:
        # Landmarks are relative to the centered analysis crop. Draw a compact
        # diagnostic skeleton in an inset so the stored source image stays raw.
        inset_size = min(260, preview.shape[0], preview.shape[1])
        inset = np.zeros((inset_size, inset_size, 3), dtype=np.uint8)
        connections = (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
        )
        for start, end in connections:
            a = tuple(np.clip(points[start] * inset_size, 0, inset_size - 1).astype(int))
            b = tuple(np.clip(points[end] * inset_size, 0, inset_size - 1).astype(int))
            cv2.line(inset, a, b, (30, 210, 250), 2)
        for point in points:
            center = tuple(np.clip(point * inset_size, 0, inset_size - 1).astype(int))
            cv2.circle(inset, center, 3, (255, 255, 255), -1)
        preview[:inset_size, preview.shape[1] - inset_size :] = inset
    expected = details.get("captured_direction") == label
    usable = bool(
        points is not None
        and expected
        and float(details.get("axis_dominance", 0.0)) >= 0.58
        and float(details.get("pose_score", 0.0)) >= 0.30
    )
    lines = [
        f"Label: {label.upper()}  View: {view}  {captured}/{target}",
        VIEW_INSTRUCTIONS[view],
        "SPACE save | N skip bucket | ESC finish",
        (
            f"Detected: {details.get('captured_direction', 'none')}  "
            f"pose={details.get('pose_score', 0):.2f}  "
            f"axis={details.get('axis_dominance', 0):.2f}  "
            f"{'READY' if usable else 'ADJUST HAND'}"
        ),
    ]
    for index, line in enumerate(lines):
        y = 30 + 30 * index
        cv2.putText(
            preview,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (70, 255, 90) if index == 3 and usable else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return preview


def _record(
    dataset_root: Path,
    rows: list[dict[str, Any]],
    jpeg: bytes,
    points: np.ndarray,
    details: dict[str, Any],
    *,
    participant: str,
    session: str,
    split: str,
    label: str,
    view: str,
    source: str,
) -> str:
    created = datetime.now(timezone.utc)
    identity = hashlib.sha256(
        jpeg
        + f"{participant}:{session}:{label}:{view}:{created.isoformat()}".encode("utf-8")
    ).hexdigest()[:20]
    relative = Path("images") / split / participant / session / label / view / f"{identity}.jpg"
    destination = dataset_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(jpeg)
    feature = np.asarray(details["feature"], dtype=np.float32)
    row = {
        "sample_id": identity,
        "image_path": relative.as_posix(),
        "created_at_utc": created.isoformat(),
        "participant_id": participant,
        "session_id": session,
        "split": split,
        "label": label,
        "view": view,
        "source": source,
        "handedness": details["handedness"],
        "width": details["source_width"],
        "height": details["source_height"],
        "input_luma_median": f"{details['input_luma_median']:.6f}",
        "sharpness": f"{details['sharpness']:.6f}",
        "pose_score": f"{details['pose_score']:.6f}",
        "axis_dominance": f"{details['axis_dominance']:.6f}",
        "captured_direction": details["captured_direction"],
        "feature_vector_json": json.dumps(feature.tolist(), separators=(",", ":")),
        "landmarks_json": json.dumps(
            np.asarray(points, dtype=np.float32).tolist(), separators=(",", ":")
        ),
    }
    rows.append(row)
    _write_manifest(dataset_root / "manifest.csv", rows)
    return identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture real multi-angle directional gesture images."
    )
    parser.add_argument("--participant", required=True, help="Anonymous stable participant ID")
    parser.add_argument(
        "--session",
        default=datetime.now(timezone.utc).strftime("session-%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--source", choices=["ncm", "webcam"], default="ncm")
    parser.add_argument("--webcam-device", type=int, default=0)
    parser.add_argument(
        "--ncm-url", default="http://127.0.0.1:8200/api/ncm/frame.jpg"
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/multiview"))
    parser.add_argument("--samples-per-view", type=int, default=8)
    parser.add_argument("--labels", nargs="+", choices=DIRECTION_LABELS, default=DIRECTION_LABELS)
    parser.add_argument("--views", nargs="+", choices=VIEWPOINTS, default=VIEWPOINTS)
    parser.add_argument("--split", choices=["auto", *CAPTURE_SPLITS], default="auto")
    return parser.parse_args()


def main() -> None:
    import cv2

    args = parse_args()
    if args.samples_per_view < 1:
        raise ValueError("--samples-per-view must be positive")
    participant = _safe_identifier(args.participant, "participant")
    session = _safe_identifier(args.session, "session")
    dataset_root = args.dataset.resolve()
    manifest_path = dataset_root / "manifest.csv"
    rows = _read_manifest(manifest_path)
    existing = _existing_split(rows, participant)
    split = existing or (
        participant_split(participant) if args.split == "auto" else args.split
    )
    if existing and args.split != "auto" and args.split != existing:
        raise ValueError(
            f"Participant {participant!r} is already assigned to {existing}; "
            f"cannot capture into {args.split}"
        )
    detector = HandDetector(Path("models/hand_landmarker.task"))
    if not detector.ready:
        raise RuntimeError(detector.error or "MediaPipe hand detector is unavailable")
    config = load_runtime_config(Path("models"))
    source = (
        NcmHttpSource(args.ncm_url)
        if args.source == "ncm"
        else WebcamSource(args.webcam_device)
    )
    print(
        f"Capturing participant={participant} session={session} split={split} "
        f"from {args.source}. Images remain local under {dataset_root}."
    )
    last_frame_id = None
    try:
        for label in args.labels:
            for view in args.views:
                existing_count = sum(
                    row["participant_id"] == participant
                    and row["session_id"] == session
                    and row["label"] == label
                    and row["view"] == view
                    for row in rows
                )
                count = existing_count
                while count < args.samples_per_view:
                    jpeg, frame_id = source.read()
                    if frame_id and frame_id == last_frame_id:
                        time.sleep(0.02)
                        continue
                    last_frame_id = frame_id
                    image, points, details = _prepare_frame(
                        jpeg, detector, config.roi_size_ratio
                    )
                    preview = _draw_preview(
                        image,
                        points,
                        details,
                        label=label,
                        view=view,
                        captured=count,
                        target=args.samples_per_view,
                    )
                    cv2.imshow("Multi-view directional capture", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        print("Capture stopped cleanly.")
                        return
                    if key in (ord("n"), ord("N")):
                        print(f"Skipped remaining {label}/{view} samples.")
                        break
                    if key != 32:
                        continue
                    usable = bool(
                        points is not None
                        and details.get("captured_direction") == label
                        and float(details.get("axis_dominance", 0.0)) >= 0.58
                        and float(details.get("pose_score", 0.0)) >= 0.30
                    )
                    if not usable:
                        print(
                            f"Not saved: expected {label}; detector saw "
                            f"{details.get('captured_direction', 'no hand')} with "
                            f"pose={details.get('pose_score', 0):.2f}, "
                            f"axis={details.get('axis_dominance', 0):.2f}."
                        )
                        continue
                    sample_id = _record(
                        dataset_root,
                        rows,
                        jpeg,
                        points,
                        details,
                        participant=participant,
                        session=session,
                        split=split,
                        label=label,
                        view=view,
                        source=args.source,
                    )
                    count += 1
                    print(f"Saved {label}/{view} {count}/{args.samples_per_view}: {sample_id}")
        print(f"Capture complete. Manifest: {manifest_path}")
    finally:
        source.close()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
