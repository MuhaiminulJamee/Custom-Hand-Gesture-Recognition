"""Multi-view capture ingestion, balancing, training, and qualification helpers.

The production classifier consumes 76-D MediaPipe landmark features rather than
camera pixels.  Real images are nevertheless retained so that each feature row
can be audited and re-extracted with a future detector.  Participant-level
splits are enforced before any oversampling or augmentation.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import CLASS_NAMES, RuntimeConfig
from backend.geometry import landmarks_to_feature
from research.v18_20 import evaluate, export_onnx, predict_onnx


DIRECTION_LABELS = ("left", "right", "up", "down")
CAPTURE_SPLITS = ("train", "val", "test")
VIEWPOINTS = (
    "front",
    "yaw_left",
    "yaw_right",
    "pitch_toward",
    "pitch_away",
    "roll_left",
    "roll_right",
    "casual",
)
MANIFEST_FIELDS = (
    "sample_id",
    "image_path",
    "created_at_utc",
    "participant_id",
    "session_id",
    "split",
    "label",
    "view",
    "source",
    "handedness",
    "width",
    "height",
    "input_luma_median",
    "sharpness",
    "pose_score",
    "axis_dominance",
    "captured_direction",
    "feature_vector_json",
    "landmarks_json",
)


def participant_split(participant_id: str, salt: str = "gesture-multiview-v1") -> str:
    """Assign a participant deterministically to a 70/15/15 split."""
    normalized = str(participant_id).strip()
    if not normalized:
        raise ValueError("participant_id cannot be empty")
    digest = hashlib.sha256(f"{salt}:{normalized}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "val"
    return "test"


def direction_from_landmarks(landmarks: np.ndarray) -> tuple[str, float]:
    """Resolve the NCM semantic direction and its axis dominance."""
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (21, 2) or not np.isfinite(points).all():
        raise ValueError("landmarks must be a finite 21x2 array")
    vector = points[8] - points[5]
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("index-finger direction is degenerate")
    dx, dy = map(float, vector)
    direction = (
        ("down" if dy > 0 else "up")
        if abs(dy) >= abs(dx)
        else ("left" if dx > 0 else "right")
    )
    return direction, float(np.max(np.abs(vector)) / norm)


def _empty_capture() -> dict[str, np.ndarray]:
    return {
        "X": np.empty((0, 76), dtype=np.float32),
        "landmarks": np.empty((0, 21, 2), dtype=np.float32),
        "y": np.empty((0,), dtype=np.int64),
        "sample_id": np.empty((0,), dtype=str),
        "participant_id": np.empty((0,), dtype=str),
        "session_id": np.empty((0,), dtype=str),
        "split": np.empty((0,), dtype=str),
        "label": np.empty((0,), dtype=str),
        "view": np.empty((0,), dtype=str),
        "image_path": np.empty((0,), dtype=str),
    }


def _json_array(value: str, shape: tuple[int, ...], description: str) -> np.ndarray:
    try:
        array = np.asarray(json.loads(value), dtype=np.float32)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {description} JSON") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{description} must be a finite array with shape {shape}")
    return array


def load_capture_manifest(
    dataset_root: Path,
    manifest_path: Path | None = None,
    *,
    require_images: bool = True,
) -> dict[str, np.ndarray]:
    """Load and strictly validate the reviewed capture manifest."""
    root = Path(dataset_root).resolve()
    manifest = Path(manifest_path).resolve() if manifest_path else root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Multi-view manifest not found: {manifest}. Run the capture script first."
        )
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in MANIFEST_FIELDS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Capture manifest is missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        return _empty_capture()

    samples: list[str] = []
    features: list[np.ndarray] = []
    landmarks: list[np.ndarray] = []
    labels: list[str] = []
    participants: list[str] = []
    sessions: list[str] = []
    splits: list[str] = []
    views: list[str] = []
    image_paths: list[str] = []
    seen_samples: set[str] = set()
    participant_splits: dict[str, str] = {}

    for line_number, row in enumerate(rows, start=2):
        sample_id = str(row["sample_id"]).strip()
        participant = str(row["participant_id"]).strip()
        session = str(row["session_id"]).strip()
        split = str(row["split"]).strip()
        label = str(row["label"]).strip()
        view = str(row["view"]).strip()
        if not sample_id or sample_id in seen_samples:
            raise ValueError(f"Manifest line {line_number} has an empty or duplicate sample_id")
        if not participant or not session:
            raise ValueError(f"Manifest line {line_number} needs participant_id and session_id")
        if split not in CAPTURE_SPLITS:
            raise ValueError(f"Manifest line {line_number} has invalid split {split!r}")
        if label not in DIRECTION_LABELS:
            raise ValueError(
                f"Manifest line {line_number} has label {label!r}; this workflow accepts "
                f"only {', '.join(DIRECTION_LABELS)}"
            )
        if view not in VIEWPOINTS:
            raise ValueError(f"Manifest line {line_number} has unknown view {view!r}")
        previous_split = participant_splits.setdefault(participant, split)
        if previous_split != split:
            raise ValueError(
                f"Participant {participant!r} appears in both {previous_split} and {split}; "
                "participant leakage is not allowed"
            )

        relative_image = Path(str(row["image_path"]).strip())
        if relative_image.is_absolute():
            raise ValueError(f"Manifest line {line_number} image_path must be relative")
        image = (root / relative_image).resolve()
        try:
            image.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Manifest line {line_number} image_path escapes the dataset") from error
        if require_images and not image.is_file():
            raise FileNotFoundError(f"Manifest line {line_number} image is missing: {image}")

        feature = _json_array(row["feature_vector_json"], (76,), "feature vector")
        points = _json_array(row["landmarks_json"], (21, 2), "landmarks")
        reconstructed = landmarks_to_feature(points)
        if not np.allclose(reconstructed, feature, atol=2e-5, rtol=2e-5):
            raise ValueError(
                f"Manifest line {line_number} feature vector does not match its landmarks"
            )

        expected_direction, _ = direction_from_landmarks(points)
        recorded_direction = str(row["captured_direction"]).strip()
        if recorded_direction != expected_direction or expected_direction != label:
            raise ValueError(
                f"Manifest line {line_number} label/direction mismatch: "
                f"label={label}, recorded={recorded_direction}, derived={expected_direction}"
            )

        seen_samples.add(sample_id)
        samples.append(sample_id)
        features.append(feature)
        landmarks.append(points)
        labels.append(label)
        participants.append(participant)
        sessions.append(session)
        splits.append(split)
        views.append(view)
        image_paths.append(relative_image.as_posix())

    return {
        "X": np.asarray(features, dtype=np.float32),
        "landmarks": np.asarray(landmarks, dtype=np.float32),
        "y": np.asarray([CLASS_NAMES.index(label) for label in labels], dtype=np.int64),
        "sample_id": np.asarray(samples),
        "participant_id": np.asarray(participants),
        "session_id": np.asarray(sessions),
        "split": np.asarray(splits),
        "label": np.asarray(labels),
        "view": np.asarray(views),
        "image_path": np.asarray(image_paths),
    }


def capture_subset(data: dict[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    if split not in CAPTURE_SPLITS:
        raise ValueError(f"Unknown capture split: {split}")
    mask = np.asarray(data["split"]) == split
    return {name: np.asarray(values)[mask] for name, values in data.items()}


def coverage_audit(
    data: dict[str, np.ndarray],
    requirements: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """Check row and participant coverage for every direction and split.

    Requirement values are ``(minimum rows per label, minimum participants per
    label, minimum viewpoints per label)``.  Two-value tuples used by tests or
    callers retain a one-view minimum. These are release floors, not claims of
    universal coverage.
    """
    requirements = requirements or {
        "train": (60, 3, 6),
        "val": (20, 2, 4),
        "test": (20, 2, 4),
    }
    issues: list[str] = []
    report: dict[str, Any] = {}
    labels = np.asarray(data["label"])
    splits = np.asarray(data["split"])
    participants = np.asarray(data["participant_id"])
    views = np.asarray(data["view"])
    for split in CAPTURE_SPLITS:
        values = requirements[split]
        if len(values) not in {2, 3}:
            raise ValueError(
                f"Coverage requirements for {split} must have two or three values"
            )
        minimum_rows, minimum_participants = values[:2]
        minimum_views = values[2] if len(values) == 3 else 1
        split_report: dict[str, Any] = {}
        for label in DIRECTION_LABELS:
            mask = (splits == split) & (labels == label)
            count = int(mask.sum())
            people = int(len(set(participants[mask].tolist())))
            seen_views = sorted(set(views[mask].tolist()))
            passed = (
                count >= minimum_rows
                and people >= minimum_participants
                and len(seen_views) >= minimum_views
            )
            split_report[label] = {
                "rows": count,
                "participants": people,
                "views": seen_views,
                "minimum_rows": minimum_rows,
                "minimum_participants": minimum_participants,
                "minimum_views": minimum_views,
                "passed": passed,
            }
            if not passed:
                issues.append(
                    f"{split}/{label}: {count} rows from {people} participants; "
                    f"{len(seen_views)} viewpoints; need at least {minimum_rows} rows "
                    f"from {minimum_participants} participants across {minimum_views} viewpoints"
                )
        report[split] = split_report
    return {
        "passed": not issues,
        "total_rows": int(len(labels)),
        "participants": int(len(set(participants.tolist()))),
        "requirements": {
            split: {
                "rows_per_label": values[0],
                "participants_per_label": values[1],
                "viewpoints_per_label": values[2] if len(values) == 3 else 1,
            }
            for split, values in requirements.items()
        },
        "splits": report,
        "issues": issues,
    }


def _resample_indices(
    available: np.ndarray, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=bool)
    if not len(available):
        raise ValueError("Cannot sample from an empty class")
    shuffled = rng.permutation(available)
    selected = np.resize(shuffled, count).astype(np.int64)
    repeated = np.arange(count) >= len(available)
    return selected, repeated


def _augment_directional_landmarks(
    points: np.ndarray, label: str, rng: np.random.Generator
) -> tuple[np.ndarray, bool]:
    original = np.asarray(points, dtype=np.float32)
    origin = original[0].copy()
    centered = original - origin
    palm_scale = max(float(np.linalg.norm(centered[9])), 1e-6)
    for _ in range(8):
        angle = float(rng.uniform(-0.10, 0.10))
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
        shear = np.asarray(
            [[1.0, rng.uniform(-0.035, 0.035)], [rng.uniform(-0.035, 0.035), 1.0]],
            dtype=np.float32,
        )
        candidate = centered @ rotation.T @ shear.T
        candidate *= rng.uniform(0.94, 1.06, size=(1, 2))
        candidate += rng.normal(0.0, palm_scale * 0.004, candidate.shape)
        candidate += origin
        try:
            direction, dominance = direction_from_landmarks(candidate)
            landmarks_to_feature(candidate)
        except ValueError:
            continue
        if direction == label and dominance >= 0.58:
            return candidate.astype(np.float32), True
    return original.copy(), False


def make_balanced_training_set(
    base: dict[str, np.ndarray],
    captured_train: dict[str, np.ndarray],
    *,
    per_class: int = 4096,
    capture_share: float = 0.25,
    maximum_capture_repeats: int = 4,
    seed: int = 1900,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Mix reviewed commands and hard negatives into the base training set."""
    if per_class < 1:
        raise ValueError("per_class must be positive")
    if not 0.0 <= capture_share <= 1.0:
        raise ValueError("capture_share must be between 0 and 1")
    if maximum_capture_repeats < 1:
        raise ValueError("maximum_capture_repeats must be positive")
    base_x = np.asarray(base["X"], dtype=np.float32)
    base_y = np.asarray(base["y"], dtype=np.int64)
    if base_x.ndim != 2 or base_x.shape[1] != 76 or base_y.shape != (len(base_x),):
        raise ValueError("Base training data must contain aligned X[N,76] and y[N]")
    capture_x = np.asarray(captured_train["X"], dtype=np.float32)
    capture_y = np.asarray(captured_train["y"], dtype=np.int64)
    capture_landmarks = np.asarray(captured_train["landmarks"], dtype=np.float32)
    if capture_x.shape != (len(capture_y), 76) or capture_landmarks.shape != (
        len(capture_y), 21, 2
    ):
        raise ValueError("Captured training arrays are not aligned")

    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    parents: list[str] = []
    origins: list[str] = []
    augmented: list[bool] = []
    audit_rows: list[dict[str, Any]] = []
    base_parents = np.asarray(
        base.get("parent_id", np.asarray([f"base:{index}" for index in range(len(base_x))]))
    )
    capture_parents = np.asarray(captured_train["sample_id"])

    for label_index in range(9):
        name = CLASS_NAMES[label_index] if label_index < len(CLASS_NAMES) else "no_gesture"
        base_available = np.flatnonzero(base_y == label_index)
        capture_available = np.flatnonzero(capture_y == label_index)
        requested_capture = round(per_class * capture_share) if len(capture_available) else 0
        capture_count = min(
            requested_capture, len(capture_available) * maximum_capture_repeats
        )
        base_count = per_class - capture_count
        base_selected, _ = _resample_indices(base_available, base_count, rng)
        for index in base_selected:
            rows.append(base_x[index])
            labels.append(label_index)
            parents.append(str(base_parents[index]))
            origins.append("v18_20")
            augmented.append(False)

        capture_selected, repeated = _resample_indices(capture_available, capture_count, rng)
        augmented_count = 0
        for position, index in enumerate(capture_selected):
            feature = capture_x[index]
            was_augmented = False
            if repeated[position] and name in DIRECTION_LABELS:
                points, was_augmented = _augment_directional_landmarks(
                    capture_landmarks[index], name, rng
                )
                feature = landmarks_to_feature(points)
            rows.append(feature)
            labels.append(label_index)
            parents.append(str(capture_parents[index]))
            origins.append("ncm_multiview")
            augmented.append(was_augmented)
            augmented_count += int(was_augmented)
        audit_rows.append(
            {
                "label": name,
                "base_rows": base_count,
                "captured_source_rows": int(len(capture_available)),
                "captured_training_rows": capture_count,
                "captured_augmented_rows": augmented_count,
                "total_rows": per_class,
            }
        )

    order = rng.permutation(len(rows))
    result = {
        "X": np.asarray(rows, dtype=np.float32)[order],
        "y": np.asarray(labels, dtype=np.int64)[order],
        "parent_id": np.asarray(parents)[order],
        "origin": np.asarray(origins)[order],
        "augmented": np.asarray(augmented, dtype=bool)[order],
    }
    return result, {
        "per_class": per_class,
        "capture_share_limit": capture_share,
        "maximum_capture_repeats": maximum_capture_repeats,
        "seed": seed,
        "classes": audit_rows,
    }


def train_candidate(
    project_root: Path,
    training: dict[str, np.ndarray],
    public_validation: dict[str, np.ndarray],
    captured_validation: dict[str, np.ndarray],
    destination: Path,
    *,
    epochs: int = 80,
    seed: int = 1900,
) -> list[dict[str, float | int]]:
    """Train with validation-only model selection and export an ONNX candidate."""
    from sklearn.metrics import f1_score, log_loss
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    x = np.asarray(training["X"], dtype=np.float32)
    y = np.asarray(training["y"], dtype=np.int64)
    public_x = np.asarray(public_validation["X"], dtype=np.float32)
    public_y = np.asarray(public_validation["y"], dtype=np.int64)
    capture_x = np.asarray(captured_validation["X"], dtype=np.float32)
    capture_y = np.asarray(captured_validation["y"], dtype=np.int64)
    scaler = StandardScaler().fit(x)
    train_x = scaler.transform(x).astype(np.float32)
    public_scaled = scaler.transform(public_x).astype(np.float32)
    capture_scaled = (
        scaler.transform(capture_x).astype(np.float32)
        if len(capture_x)
        else np.empty((0, 76), dtype=np.float32)
    )
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        alpha=0.003,
        batch_size=256,
        learning_rate_init=0.0007,
        random_state=seed,
        max_iter=1,
    )
    best = None
    best_score = -1.0
    stale = 0
    history: list[dict[str, float | int]] = []
    with threadpool_limits(limits=1):
        for epoch in range(epochs):
            model.partial_fit(train_x, y, classes=np.arange(9))
            public_probabilities = model.predict_proba(public_scaled)
            public_f1 = float(
                f1_score(
                    public_y,
                    public_probabilities.argmax(1),
                    labels=np.arange(9),
                    average="macro",
                    zero_division=0,
                )
            )
            if len(capture_scaled):
                capture_probabilities = model.predict_proba(capture_scaled)
                capture_f1 = float(
                    f1_score(
                        capture_y,
                        capture_probabilities.argmax(1),
                        labels=np.unique(capture_y),
                        average="macro",
                        zero_division=0,
                    )
                )
                selection_score = 0.60 * public_f1 + 0.40 * capture_f1
            else:
                capture_f1 = float("nan")
                selection_score = public_f1
            row: dict[str, float | int] = {
                "epoch": epoch + 1,
                "train_loss": float(model.loss_),
                "public_val_log_loss": float(
                    log_loss(public_y, public_probabilities, labels=np.arange(9))
                ),
                "public_val_macro_f1": public_f1,
                "captured_val_directional_macro_f1": capture_f1,
                "selection_score": selection_score,
            }
            history.append(row)
            if selection_score > best_score + 0.0001:
                best = copy.deepcopy(model)
                best_score = selection_score
                stale = 0
            else:
                stale += 1
            if stale >= 15:
                break
    if best is None:
        raise RuntimeError("Training did not produce a candidate model")
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(project_root, best, scaler, destination)
    return history


def _public_metrics(
    model_path: Path, data: dict[str, np.ndarray], config: RuntimeConfig
) -> dict[str, Any]:
    probabilities, mass = predict_onnx(model_path, data["X"])
    result = evaluate(probabilities, mass, data, config=config)
    per_class = {
        name: {
            "precision": float(result["report"][name]["precision"]),
            "recall": float(result["report"][name]["recall"]),
            "f1": float(result["report"][name]["f1-score"]),
            "support": int(result["report"][name]["support"]),
        }
        for name in (*CLASS_NAMES, "no_gesture")
    }
    return {"metrics": result["metrics"], "per_class": per_class}


def _captured_metrics(
    model_path: Path, data: dict[str, np.ndarray], config: RuntimeConfig
) -> dict[str, Any]:
    if not len(data["X"]):
        return {
            "rows": 0,
            "directional_macro_f1": 0.0,
            "correct_command_rate": 0.0,
            "known_rejection_rate": 1.0,
            "minimum_directional_recall": 0.0,
            "per_class_recall": {name: 0.0 for name in DIRECTION_LABELS},
        }
    from sklearn.metrics import f1_score, recall_score

    probabilities, mass = predict_onnx(model_path, data["X"])
    result = evaluate(probabilities, mass, data, config=config)
    truth = np.asarray(result["truth"], dtype=np.int64)
    predicted = np.asarray(result["predicted"], dtype=np.int64)
    recalls = recall_score(
        truth, predicted, labels=np.arange(4), average=None, zero_division=0
    )
    correct = predicted == truth
    return {
        "rows": int(len(truth)),
        "directional_macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=np.arange(4),
                average="macro",
                zero_division=0,
            )
        ),
        "correct_command_rate": float(np.mean(correct)),
        "known_rejection_rate": float(np.mean(predicted == 8)),
        "minimum_directional_recall": float(np.min(recalls)),
        "per_class_recall": {
            name: float(recalls[index]) for index, name in enumerate(DIRECTION_LABELS)
        },
    }


def qualify_candidate(
    baseline_path: Path,
    candidate_path: Path,
    public_test: dict[str, np.ndarray],
    captured_test: dict[str, np.ndarray],
    config: RuntimeConfig,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """Compare the candidate with production on old and new untouched tests."""
    baseline_public = _public_metrics(baseline_path, public_test, config)
    candidate_public = _public_metrics(candidate_path, public_test, config)
    baseline_capture = _captured_metrics(baseline_path, captured_test, config)
    candidate_capture = _captured_metrics(candidate_path, captured_test, config)
    base_public = baseline_public["metrics"]
    new_public = candidate_public["metrics"]
    gates = {
        "coverage_passed": bool(coverage.get("passed")),
        "public_runtime_macro_f1_preserved": bool(
            new_public["runtime_macro_f1"] >= base_public["runtime_macro_f1"] - 0.005
        ),
        "public_correct_command_rate_preserved": bool(
            new_public["correct_command_rate"] >= base_public["correct_command_rate"] - 0.01
        ),
        "public_unknown_false_accept_rate": bool(
            new_public["unknown_false_accept_rate"]
            <= max(0.02, base_public["unknown_false_accept_rate"] + 0.005)
        ),
        "captured_directional_macro_f1": bool(
            candidate_capture["directional_macro_f1"] >= 0.90
        ),
        "captured_correct_command_rate": bool(
            candidate_capture["correct_command_rate"] >= 0.85
        ),
        "captured_minimum_directional_recall": bool(
            candidate_capture["minimum_directional_recall"] >= 0.80
        ),
        "captured_not_worse_than_baseline": bool(
            candidate_capture["correct_command_rate"]
            >= baseline_capture["correct_command_rate"] - 0.01
        ),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "baseline": {"public_test": baseline_public, "captured_test": baseline_capture},
        "candidate": {"public_test": candidate_public, "captured_test": candidate_capture},
        "note": (
            "Passing these software gates does not replace a live NCM action test. "
            "The untouched participant test split must remain frozen."
        ),
    }


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        raise ValueError("Training history is empty")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
