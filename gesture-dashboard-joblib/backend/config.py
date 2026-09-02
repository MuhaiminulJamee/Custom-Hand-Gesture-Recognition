from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIRECTORY = Path(os.getenv("GESTURE_MODELS_DIR", PROJECT_ROOT / "models"))
FEEDBACK_DIRECTORY = Path(os.getenv("GESTURE_FEEDBACK_DIR", PROJECT_ROOT / "feedback"))


def _environment_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false, 1/0, yes/no, or on/off.")


def _environment_integer(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value

CLASS_NAMES = [
    "call", "rock", "like", "ok", "one", "one_down", "one_left",
    "one_right", "palm", "peace", "dorsal_hand", "fist", "zoom_in",
    "zoom_out", "no_gesture",
]

GESTURE_TO_ACTION = {
    "call": "Detect My Face",
    "rock": "Pause Video",
    "like": "Stop / Continue",
    "ok": "Start Recording",
    "one": "Move Up",
    "one_down": "Move Down",
    "one_left": "Move Left",
    "one_right": "Move Right",
    "palm": "Open Palm",
    "peace": "End Recording",
    "dorsal_hand": "Return to Main Position",
    "fist": "Fist (Follow Object phase)",
    "zoom_in": "Zoom In",
    "zoom_out": "Zoom Out",
    "no_gesture": "No Gesture",
}

FEATURE_NAMES = (
    [f"landmark_{index}_{axis}" for index in range(21) for axis in ("x", "y")]
    + [f"landmark_{index}_radius" for index in range(21)]
    + ["index_segment_dx", "index_segment_dy", "wrist_index_dx", "wrist_index_dy"]
    + ["thumb_index_gap_ratio"]
    + [
        "thumb_extension", "index_extension", "middle_extension",
        "ring_extension", "pinky_extension", "other_fingers_folded",
        "thumb_tip_index_mcp_ratio", "thumb_reach_ratio",
    ]
)
assert len(FEATURE_NAMES) == 76


@dataclass(frozen=True, slots=True)
class ProductionSettings:
    """Operational controls that do not alter classifier behavior."""

    environment: str
    require_artifact_manifest: bool
    allow_force_learning: bool
    allow_artifact_reload: bool
    admin_token: str
    max_image_bytes: int
    max_websocket_frame_bytes: int
    max_active_sessions: int
    metrics_window_size: int
    minimum_macro_f1: float
    minimum_per_class_f1: float
    deferred_quality_classes: tuple[str, ...]
    log_directory: Path

    @property
    def production_mode(self) -> bool:
        return self.environment == "production"

    def public_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "production_mode": self.production_mode,
            "require_artifact_manifest": self.require_artifact_manifest,
            "allow_force_learning": self.allow_force_learning,
            "allow_artifact_reload": self.allow_artifact_reload,
            "admin_token_configured": bool(self.admin_token),
            "max_image_bytes": self.max_image_bytes,
            "max_websocket_frame_bytes": self.max_websocket_frame_bytes,
            "max_active_sessions": self.max_active_sessions,
            "metrics_window_size": self.metrics_window_size,
            "quality_gate": {
                "minimum_macro_f1": self.minimum_macro_f1,
                "minimum_per_class_f1": self.minimum_per_class_f1,
                "deferred_classes": list(self.deferred_quality_classes),
            },
        }


def load_production_settings() -> ProductionSettings:
    environment = os.getenv("GESTURE_ENVIRONMENT", "production").strip().lower()
    if environment not in {"development", "test", "production"}:
        raise ValueError("GESTURE_ENVIRONMENT must be development, test, or production.")
    default_manifest = environment == "production"
    log_directory = Path(
        os.getenv("GESTURE_LOG_DIR", str(PROJECT_ROOT / ".runtime" / "logs"))
    ).resolve()
    return ProductionSettings(
        environment=environment,
        require_artifact_manifest=_environment_flag(
            "GESTURE_REQUIRE_ARTIFACT_MANIFEST", default_manifest
        ),
        allow_force_learning=_environment_flag("GESTURE_ALLOW_FORCE_LEARNING", False),
        allow_artifact_reload=_environment_flag(
            "GESTURE_ALLOW_ARTIFACT_RELOAD", environment != "production"
        ),
        admin_token=os.getenv("GESTURE_ADMIN_TOKEN", "").strip(),
        max_image_bytes=_environment_integer(
            "GESTURE_MAX_IMAGE_BYTES", 8_000_000, 100_000, 25_000_000
        ),
        max_websocket_frame_bytes=_environment_integer(
            "GESTURE_MAX_WEBSOCKET_FRAME_BYTES", 4_000_000, 100_000, 15_000_000
        ),
        max_active_sessions=_environment_integer(
            "GESTURE_MAX_ACTIVE_SESSIONS", 4, 1, 32
        ),
        metrics_window_size=_environment_integer(
            "GESTURE_METRICS_WINDOW_SIZE", 1000, 50, 100_000
        ),
        minimum_macro_f1=float(os.getenv("GESTURE_MINIMUM_MACRO_F1", "0.90")),
        minimum_per_class_f1=float(
            os.getenv("GESTURE_MINIMUM_PER_CLASS_F1", "0.80")
        ),
        deferred_quality_classes=("no_gesture",),
        log_directory=log_directory,
    )


@dataclass(slots=True)
class RuntimeConfig:
    class_names: list[str] = field(default_factory=lambda: CLASS_NAMES.copy())
    gesture_to_action: dict[str, str] = field(
        default_factory=lambda: GESTURE_TO_ACTION.copy()
    )
    feature_names: list[str] = field(default_factory=lambda: FEATURE_NAMES.copy())
    target_fps: float = 10.0
    ema_alpha: float = 0.65
    confidence_floor: float = 0.70
    stable_frames_required: int = 3
    zoom_hold_seconds: float = 1.0
    action_cooldown_seconds: float = 0.80
    follow_timeout_seconds: float = 20.0
    follow_hold_seconds: float = 2.5
    follow_session_timeout_seconds: float = 120.0
    follow_success_display_seconds: float = 8.0
    follow_min_confidence: float = 0.70
    roi_size_ratio: float = 0.92
    selected_model_name: str = "MLP"
    zoom_gap_calibration: dict[str, float] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.class_names)}

    @property
    def frame_interval_ms(self) -> int:
        return max(1, round(1000 / max(self.target_fps, 0.1)))

    @property
    def frame_budget_ms(self) -> float:
        return 1000.0 / max(self.target_fps, 0.1)

    def public_dict(self) -> dict[str, Any]:
        return {
            "class_names": self.class_names,
            "gesture_to_action": self.gesture_to_action,
            "feature_count": len(self.feature_names),
            "target_fps": self.target_fps,
            "frame_interval_ms": self.frame_interval_ms,
            "frame_budget_ms": self.frame_budget_ms,
            "ema_alpha": self.ema_alpha,
            "confidence_floor": self.confidence_floor,
            "stable_frames_required": self.stable_frames_required,
            "zoom_hold_seconds": self.zoom_hold_seconds,
            "action_cooldown_seconds": self.action_cooldown_seconds,
            "follow_timeout_seconds": self.follow_timeout_seconds,
            "follow_hold_seconds": self.follow_hold_seconds,
            "follow_session_timeout_seconds": self.follow_session_timeout_seconds,
            "follow_success_display_seconds": self.follow_success_display_seconds,
            "follow_min_confidence": self.follow_min_confidence,
            "roi_size_ratio": self.roi_size_ratio,
            "runtime_format": "Joblib",
            "selected_model_name": self.selected_model_name,
            "zoom_gap_calibration": self.zoom_gap_calibration,
            "follow_object_sequence": ["palm", "fist", "palm"],
            "follow_object_base_models": ["SVM", "MLP"],
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_runtime_config(models_directory: Path = MODELS_DIRECTORY) -> RuntimeConfig:
    path = models_directory / "gesture_joblib_runtime_config.json"
    data = _read_json(path)
    smoothing = data.get("temporal_smoothing", {})
    zoom = data.get("zoom_resolution", {})
    class_names = list(data.get("class_names") or CLASS_NAMES)
    if class_names != CLASS_NAMES:
        raise ValueError(
            "Runtime configuration class order does not match the notebook's "
            f"15-class taxonomy: {class_names}"
        )
    feature_names = list(data.get("feature_names") or FEATURE_NAMES)
    if feature_names != FEATURE_NAMES:
        raise ValueError("Runtime configuration does not contain the exact 76-D feature order.")
    return RuntimeConfig(
        class_names=class_names,
        gesture_to_action=dict(data.get("gesture_to_action") or GESTURE_TO_ACTION),
        feature_names=feature_names,
        target_fps=float(data.get("target_fps", 10.0)),
        ema_alpha=float(smoothing.get("alpha", 0.65)),
        confidence_floor=float(smoothing.get("confidence_floor", 0.70)),
        stable_frames_required=int(smoothing.get("stable_frames_required", 3)),
        zoom_hold_seconds=float(zoom.get("hold_seconds", 1.0)),
        follow_timeout_seconds=float(
            data.get("follow_object_step_timeout_seconds", 20.0)
        ),
        follow_hold_seconds=float(data.get("follow_object_hold_seconds", 2.5)),
        follow_session_timeout_seconds=float(
            data.get("follow_object_session_timeout_seconds", 120.0)
        ),
        follow_success_display_seconds=float(
            data.get("follow_object_success_display_seconds", 8.0)
        ),
        follow_min_confidence=float(
            data.get("follow_object_base_consensus_min_confidence", 0.70)
        ),
        roi_size_ratio=float(data.get("roi_size_ratio", 0.92)),
        selected_model_name=str(
            data.get("validation_selected_primary_model")
            or "MLP"
        ),
        zoom_gap_calibration=data.get("zoom_gap_calibration"),
        raw=data,
    )
