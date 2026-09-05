from __future__ import annotations

"""Build and qualify the eight-command ONNX deployment graph.

The supplied 15-way MLP contains useful hard-negative knowledge.  This script
keeps that knowledge inside the graph, exposes only the requested eight
commands, and exports the retained probability mass as an open-set score.  The
untouched test cache is used once, after configuration choices are made, to
produce the release report.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


SOURCE_CLASSES = [
    "call", "rock", "like", "ok", "one", "one_down", "one_left",
    "one_right", "palm", "peace", "dorsal_hand", "fist", "zoom_in",
    "zoom_out", "no_gesture",
]
CLASS_MAPPING = [
    ("left", "one_left"),
    ("right", "one_right"),
    ("up", "one"),
    ("down", "one_down"),
    ("open_palm", "palm"),
    ("like", "like"),
    ("dorsal", "dorsal_hand"),
    ("ok", "ok"),
]
TARGET_CLASSES = [target for target, _ in CLASS_MAPPING]
SOURCE_INDEXES = np.asarray(
    [SOURCE_CLASSES.index(source) for _, source in CLASS_MAPPING], dtype=np.int64
)
GESTURE_TO_ACTION = {
    "left": "Move Left",
    "right": "Move Right",
    "up": "Move Up",
    "down": "Move Down",
    "open_palm": "Enable / Disable Object Tracking",
    "like": "Play / Pause",
    "dorsal": "Return to Default Position",
    "ok": "Start / Stop Recording",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_contract(source_model: Path, source_config: Path) -> None:
    if not source_model.is_file():
        raise FileNotFoundError(f"Source ONNX model is missing: {source_model}")
    if not source_config.is_file():
        raise FileNotFoundError(f"Source runtime config is missing: {source_config}")
    data = json.loads(source_config.read_text(encoding="utf-8"))
    if list(data.get("class_names") or []) != SOURCE_CLASSES:
        raise ValueError("Source class order does not match the qualified 15-way model.")
    if len(list(data.get("feature_names") or [])) != 76:
        raise ValueError("Source model does not declare the required 76-D feature contract.")


def _producer_map(nodes: Iterable[object]) -> dict[str, object]:
    return {
        output: node
        for node in nodes
        for output in getattr(node, "output", [])
        if output
    }


def build_model(source_path: Path, destination_path: Path) -> tuple[bytes, str]:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    source_bytes = source_path.read_bytes()
    model = onnx.load_from_string(source_bytes)
    probability_output = next(
        (output for output in model.graph.output if output.name == "probabilities"),
        None,
    )
    if probability_output is None:
        raise ValueError("Source ONNX model has no 'probabilities' output.")
    dimensions = probability_output.type.tensor_type.shape.dim
    if len(dimensions) != 2 or dimensions[1].dim_value != len(SOURCE_CLASSES):
        raise ValueError("Source ONNX model is not the expected 15-output graph.")

    producers = _producer_map(model.graph.node)
    output_node = producers.get("probabilities")
    if output_node is None or len(output_node.input) != 1:
        raise ValueError("Could not locate the source probability tensor.")
    all_probabilities = output_node.input[0]

    required_nodes: set[int] = set()

    def require_tensor(name: str) -> None:
        node = producers.get(name)
        if node is None:
            return
        identity = id(node)
        if identity in required_nodes:
            return
        required_nodes.add(identity)
        for input_name in node.input:
            require_tensor(input_name)

    require_tensor(all_probabilities)
    kept = [node for node in model.graph.node if id(node) in required_nodes]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    used_initializers = {name for node in kept for name in node.input}
    retained_initializers = [
        value for value in model.graph.initializer if value.name in used_initializers
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)

    model.graph.initializer.extend([
        numpy_helper.from_array(SOURCE_INDEXES, name="eight_class_source_indexes"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="eight_class_sum_axis"),
        numpy_helper.from_array(
            np.asarray([1e-30], dtype=np.float32), name="eight_class_mass_epsilon"
        ),
    ])
    model.graph.node.extend([
        helper.make_node(
            "Gather",
            [all_probabilities, "eight_class_source_indexes"],
            ["selected_probabilities"],
            axis=1,
            name="SelectEightGestures",
        ),
        helper.make_node(
            "ReduceSum",
            ["selected_probabilities", "eight_class_sum_axis"],
            ["known_gesture_mass"],
            keepdims=1,
            name="KnownGestureMass",
        ),
        helper.make_node(
            "Add",
            ["selected_probabilities", "eight_class_mass_epsilon"],
            ["stabilized_probabilities"],
            name="StabilizeEightGestureProbabilities",
        ),
        helper.make_node(
            "ReduceSum",
            ["stabilized_probabilities", "eight_class_sum_axis"],
            ["safe_known_gesture_mass"],
            keepdims=1,
            name="SafeKnownGestureDenominator",
        ),
        helper.make_node(
            "Div",
            ["stabilized_probabilities", "safe_known_gesture_mass"],
            ["probabilities"],
            name="NormalizeEightGestures",
        ),
        helper.make_node(
            "ArgMax",
            ["probabilities"],
            ["label"],
            axis=1,
            keepdims=0,
            name="EightGestureLabel",
        ),
    ])
    del model.graph.output[:]
    model.graph.output.extend([
        helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
        helper.make_tensor_value_info(
            "probabilities", TensorProto.FLOAT, [None, len(TARGET_CLASSES)]
        ),
        helper.make_tensor_value_info("known_gesture_mass", TensorProto.FLOAT, [None, 1]),
    ])
    model.graph.name = "gesture_mlp_eight_commands_with_open_set_rejection"
    model.doc_string = (
        "Eight-command conditional classifier. known_gesture_mass preserves the "
        "source hard-negative evidence and is not a ninth gesture class."
    )
    onnx.checker.check_model(model)
    encoded = model.SerializeToString()
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(destination_path)
    return source_bytes, hashlib.sha256(encoded).hexdigest()


def _runtime_outputs(model_path: Path, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    probabilities, mass = session.run(
        ["probabilities", "known_gesture_mass"],
        {input_name: np.asarray(features, dtype=np.float32)},
    )
    return np.asarray(probabilities, dtype=np.float64), np.asarray(mass).reshape(-1)


def _source_outputs(source_bytes: bytes, features: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(source_bytes, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(
        ["probabilities"], {input_name: np.asarray(features, dtype=np.float32)}
    )[0]
    selected = np.asarray(output, dtype=np.float64)[:, SOURCE_INDEXES]
    return selected / np.maximum(selected.sum(axis=1, keepdims=True), 1e-12)


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        return {name: np.asarray(cache[name]) for name in cache.files}


def _mapped_labels(labels: np.ndarray) -> np.ndarray:
    mapping = {int(source): target for target, source in enumerate(SOURCE_INDEXES)}
    return np.asarray([mapping.get(int(value), -1) for value in labels], dtype=np.int64)


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def qualify(
    model_path: Path,
    source_bytes: bytes,
    cache_directory: Path,
    models_directory: Path,
    known_mass_floor: float,
) -> dict[str, object]:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    validation = _load_cache(cache_directory / "gesture_online_validation_cache.npz")
    untouched = _load_cache(cache_directory / "gesture_online_untouched_test_cache.npz")
    replay = _load_cache(cache_directory / "gesture_online_replay_cache.npz")

    parity_labels = _mapped_labels(validation["y"])
    parity_features = validation["X"][parity_labels >= 0][:256]
    expected = _source_outputs(source_bytes, parity_features)
    actual, _ = _runtime_outputs(model_path, parity_features)
    absolute_error = np.abs(expected - actual)
    parity = {
        "validation_samples": int(len(parity_features)),
        "prediction_agreement": float(np.mean(expected.argmax(axis=1) == actual.argmax(axis=1))),
        "maximum_absolute_probability_error": float(absolute_error.max()),
        "mean_absolute_probability_error": float(absolute_error.mean()),
        "minimum_agreement": 0.99,
        "maximum_error_limit": 1e-4,
    }
    parity["passed"] = bool(
        parity["prediction_agreement"] >= parity["minimum_agreement"]
        and parity["maximum_absolute_probability_error"] <= parity["maximum_error_limit"]
    )

    mapped_test_y = _mapped_labels(untouched["y"])
    known = mapped_test_y >= 0
    probabilities, mass = _runtime_outputs(model_path, untouched["X"])
    predicted = probabilities.argmax(axis=1)
    known_y = mapped_test_y[known]
    known_predicted = predicted[known]
    report = classification_report(
        known_y,
        known_predicted,
        labels=np.arange(len(TARGET_CLASSES)),
        target_names=TARGET_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    per_class = []
    for class_name in TARGET_CLASSES:
        row = report[class_name]
        per_class.append({
            "class": class_name,
            "precision": row["precision"],
            "recall": row["recall"],
            "f1_score": row["f1-score"],
            "support": int(row["support"]),
        })
    _write_csv(
        models_directory / "eight_gesture_classification_report.csv",
        ["class", "precision", "recall", "f1_score", "support"],
        per_class,
    )
    matrix = confusion_matrix(
        known_y, known_predicted, labels=np.arange(len(TARGET_CLASSES))
    )
    _write_csv(
        models_directory / "eight_gesture_confusion_matrix.csv",
        ["actual", *TARGET_CLASSES],
        [
            {"actual": name, **{target: int(matrix[row, column]) for column, target in enumerate(TARGET_CLASSES)}}
            for row, name in enumerate(TARGET_CLASSES)
        ],
    )

    accepted = mass >= known_mass_floor
    unknown = ~known
    accepted_known = accepted & known
    open_set = {
        "known_mass_floor": float(known_mass_floor),
        "known_samples": int(known.sum()),
        "unknown_samples": int(unknown.sum()),
        "known_acceptance_rate": float(accepted_known.sum() / max(1, known.sum())),
        "unknown_false_acceptance_rate": float((accepted & unknown).sum() / max(1, unknown.sum())),
        "accepted_known_accuracy": float(
            np.mean(predicted[accepted_known] == mapped_test_y[accepted_known])
        ),
    }
    source_test_y = untouched["y"].astype(np.int64)
    rock = source_test_y == SOURCE_CLASSES.index("rock")
    dorsal = source_test_y == SOURCE_CLASSES.index("dorsal_hand")
    down = source_test_y == SOURCE_CLASSES.index("one_down")
    down_index = TARGET_CLASSES.index("down")
    dorsal_index = TARGET_CLASSES.index("dorsal")
    accepted_rock = rock & accepted
    hard_cases = {
        "rock_hard_negative_samples": int(rock.sum()),
        "rock_false_acceptance_rate": float(np.mean(accepted[rock])),
        "rock_accepted_as_down_rate": float(
            np.sum(accepted_rock & (predicted == down_index)) / max(1, rock.sum())
        ),
        "dorsal_samples": int(dorsal.sum()),
        "dorsal_recall": float(np.mean(predicted[dorsal] == dorsal_index)),
        "dorsal_as_down_rate": float(np.mean(predicted[dorsal] == down_index)),
        "down_samples": int(down.sum()),
        "down_recall": float(np.mean(predicted[down] == down_index)),
        "down_as_dorsal_rate": float(np.mean(predicted[down] == dorsal_index)),
    }
    _write_csv(
        models_directory / "eight_gesture_hard_case_metrics.csv",
        list(hard_cases),
        [hard_cases],
    )
    metrics = {
        "model": "EightGestureONNX",
        "test_samples": int(known.sum()),
        "accuracy": float(np.mean(known_predicted == known_y)),
        "macro_f1": float(f1_score(known_y, known_predicted, average="macro")),
        "minimum_per_class_f1": float(min(row["f1_score"] for row in per_class)),
        **open_set,
    }
    _write_csv(
        models_directory / "eight_gesture_test_metrics.csv",
        list(metrics),
        [metrics],
    )

    # Runtime learning caches deliberately preserve hard negatives as label -1.
    for source_name, values in (
        ("gesture_online_replay_cache.npz", replay),
        ("gesture_online_validation_cache.npz", validation),
        ("gesture_online_untouched_test_cache.npz", untouched),
    ):
        payload = {
            "X": values["X"].astype(np.float32),
            "y": _mapped_labels(values["y"]),
            # Preserve the source taxonomy only as evaluation provenance. Runtime
            # and feedback code consume ``y`` (eight classes or -1) exclusively.
            "source_y": values["y"].astype(np.int64),
            "source_class_names": np.asarray(SOURCE_CLASSES),
        }
        if "landmarks" in values:
            payload["landmarks"] = values["landmarks"].astype(np.float32)
        destination = models_directory / source_name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(destination)

    return {
        "parity": parity,
        "test_metrics": metrics,
        "open_set": open_set,
        "hard_cases": hard_cases,
    }


def write_runtime_config(
    source_config_path: Path,
    destination: Path,
    qualification: dict[str, object],
) -> list[str]:
    source = json.loads(source_config_path.read_text(encoding="utf-8"))
    feature_names = list(source["feature_names"])
    config = {
        "schema_version": 2,
        "model_file": "gesture_mlp_production.onnx",
        "model_format": "ONNX float32, eight command outputs plus open-set score",
        "input_shape": [1, len(feature_names)],
        "input_feature_version": source.get("input_feature_version"),
        "feature_names": feature_names,
        "class_names": TARGET_CLASSES,
        "reject_label": "no_gesture",
        "feedback_labels": [*TARGET_CLASSES, "no_gesture"],
        "gesture_to_action": GESTURE_TO_ACTION,
        "target_fps": 20,
        "frame_interval_ms": 50,
        "num_hands": 1,
        "recommended_analysis_resolution": [320, 320],
        "roi_size_ratio": 0.92,
        "camera_orientation": {
            "mirror_horizontal": False,
            "direction_semantics": "unmirrored image coordinates; x decreases left and increases right",
        },
        "hand_detection": {
            "running_mode": "VIDEO",
            "minimum_detection_confidence": 0.55,
            "minimum_presence_confidence": 0.55,
            "minimum_tracking_confidence": 0.60,
            "minimum_hand_area_ratio": 0.018,
        },
        "low_light_enhancement": {
            "enabled": True,
            "median_luma_trigger": 92.0,
            "clahe_clip_limit_range": [1.6, 2.4],
            "gamma_range": [0.55, 1.0],
            "near_blank_mean_luma_maximum": 6.0,
        },
        "landmark_smoothing": {
            "algorithm": "velocity_adaptive_ema",
            "stationary_alpha": 0.18,
            "maximum_alpha": 0.72,
            "motion_gain": 1.35,
            "shape_jump_threshold": 0.62,
            "centroid_jump_threshold": 1.20,
            "persistent_jump_reacquire_frames": 2,
            "missing_frames_before_reset": 3,
        },
        "temporal_smoothing": {
            "algorithm": "probability_ema_with_open_set_rejection",
            "alpha": 0.45,
            "confidence_floor": 0.80,
            "probability_margin_floor": 0.18,
            "known_mass_floor": 0.80,
            "stable_frames_required": 5,
            "release_frames_required": 3,
            "minimum_hold_seconds": 0.18,
            "reject_label": "no_gesture",
        },
        "directional_resolution": {
            "pose_score": 0.50,
            "minimum_axis_dominance": 0.78,
            "maximum_mean_non_index_finger_extension": 0.70,
            "hard_geometry_veto": True,
            "horizontal_mirror": False,
        },
        "dorsal_resolution": {
            "pose_score": 0.76,
            "minimum_downward_fingers": 4,
            "minimum_extension_score": 0.50,
            "hard_geometry_veto": True,
        },
        "pose_validation": {
            "enabled": True,
            "like_minimum_thumb_extension": 0.60,
            "like_maximum_other_finger_extension": 0.58,
            "ok_maximum_thumb_index_gap_ratio": 0.58,
            "ok_minimum_extended_other_fingers": 2,
            "open_palm_minimum_extended_fingers": 4,
        },
        "action_cooldown_seconds": 0.80,
        "online_learning": {
            "enabled": True,
            "mode": "validation_gated_numpy_residual_adapter",
            "state_file": "gesture_online_adapter_state.npz",
            "replay_cache_file": "gesture_online_replay_cache.npz",
            "validation_cache_file": "gesture_online_validation_cache.npz",
            "negative_feedback_label": "no_gesture",
        },
        "qualification": qualification,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return feature_names


def write_metadata(
    destination: Path,
    model_path: Path,
    feature_names: list[str],
    qualification: dict[str, object],
) -> None:
    import onnx
    import onnxruntime

    model = onnx.load(model_path)
    metadata = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "ONNX",
        "model_name": "EightGestureMLP",
        "onnx_file": model_path.name,
        "onnx_bytes": model_path.stat().st_size,
        "onnx_sha256": sha256_file(model_path),
        "onnx_ir_version": model.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "input": {"name": "landmark_features", "dtype": "float32", "shape": [None, 76]},
        "probability_output": "probabilities",
        "known_mass_output": "known_gesture_mass",
        "output_class_order": TARGET_CLASSES,
        "source_class_mapping": {target: source for target, source in CLASS_MAPPING},
        "feature_names": feature_names,
        "parity": qualification["parity"],
        "quality": qualification["test_metrics"],
        "open_set_rejection": qualification["open_set"],
        "hard_case_metrics": qualification["hard_cases"],
        "runtime_note": (
            "Only eight command probabilities are exposed. no_gesture is a runtime "
            "rejection state based on hand presence, known mass, pose geometry, margin, "
            "and temporal stability; it is not a ninth classifier class."
        ),
        "tool_versions": {
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "numpy": np.__version__,
        },
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    default_cache = project.parent / "gesture-dashboard-joblib" / "models"
    default_source = project.parent / "gesture-dashboard-onnx" / "models"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-model",
        type=Path,
        default=default_source / "gesture_mlp_production.onnx",
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=default_source / "gesture_mobile_runtime_config.json",
    )
    parser.add_argument("--cache-directory", type=Path, default=default_cache)
    parser.add_argument("--models-directory", type=Path, default=project / "models")
    parser.add_argument("--known-mass-floor", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_model = args.source_model.resolve()
    source_config = args.source_config.resolve()
    validate_source_contract(source_model, source_config)
    args.models_directory.mkdir(parents=True, exist_ok=True)
    destination_model = args.models_directory / "gesture_mlp_production.onnx"
    source_bytes, _ = build_model(source_model, destination_model)
    qualification = qualify(
        destination_model,
        source_bytes,
        args.cache_directory.resolve(),
        args.models_directory.resolve(),
        args.known_mass_floor,
    )
    feature_names = write_runtime_config(
        source_config,
        args.models_directory / "gesture_mobile_runtime_config.json",
        qualification,
    )
    write_metadata(
        args.models_directory / "gesture_mlp_onnx_metadata.json",
        destination_model,
        feature_names,
        qualification,
    )
    print(json.dumps(qualification, indent=2))


if __name__ == "__main__":
    main()
