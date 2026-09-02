from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import MODELS_DIRECTORY, RuntimeConfig
from .follow_object import FollowObjectStateMachine
from .geometry import GeometryResolver, landmarks_to_feature
from .runtime import TemporalGate


@dataclass(slots=True)
class RuntimeSession:
    config: RuntimeConfig
    temporal_gate: TemporalGate = field(init=False)
    follow_object: FollowObjectStateMachine = field(init=False)
    diagnostic_gates: dict[str, TemporalGate] = field(default_factory=dict)
    frame_times: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    last_action_gesture: str | None = None
    last_action_at: float = 0.0

    def __post_init__(self) -> None:
        self.temporal_gate = TemporalGate(self.config)
        self.follow_object = FollowObjectStateMachine(
            timeout_seconds=self.config.follow_timeout_seconds,
            hold_seconds=self.config.follow_hold_seconds,
            session_timeout_seconds=self.config.follow_session_timeout_seconds,
        )

    def reset(self) -> None:
        self.temporal_gate.reset()
        self.follow_object.reset()
        self.diagnostic_gates.clear()
        self.frame_times.clear()
        self.last_action_gesture = None
        self.last_action_at = 0.0

    def record_frame(self, now: float) -> float:
        self.frame_times.append(now)
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / elapsed if elapsed > 1e-9 else 0.0

    def diagnostic_gate(self, model_name: str) -> TemporalGate:
        if model_name not in self.diagnostic_gates:
            self.diagnostic_gates[model_name] = TemporalGate(self.config)
        return self.diagnostic_gates[model_name]


class FollowObjectProbabilityEnsemble:
    """Equal SVM + MLP probabilities used only by the opt-in sequence."""

    def __init__(self, svm: Any, mlp: Any, class_count: int):
        self.named_models = (("SVM", svm), ("MLP", mlp))
        self.class_count = int(class_count)
        self.classes_ = np.arange(self.class_count, dtype=int)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        aligned_rows: list[np.ndarray] = []
        for _, model in self.named_models:
            raw = np.asarray(model.predict_proba(features), dtype=np.float64)
            model_classes = np.asarray(
                getattr(model, "classes_", np.arange(raw.shape[1])), dtype=int
            )
            aligned = np.zeros((len(raw), self.class_count), dtype=np.float64)
            aligned[:, model_classes] = raw
            aligned_rows.append(aligned)
        averaged = np.mean(np.stack(aligned_rows, axis=0), axis=0)
        averaged = np.clip(averaged, 1e-12, None)
        return averaged / averaged.sum(axis=1, keepdims=True)


class ModelManager:
    def __init__(
        self,
        config: RuntimeConfig,
        models_directory: Path = MODELS_DIRECTORY,
        *,
        load_models: bool = True,
    ):
        self.config = config
        self.models_directory = models_directory
        self.models: dict[str, Any] = {}
        self.selected_model_name = config.selected_model_name
        self.errors: list[str] = []
        self.bundle_metadata: dict[str, Any] = {}
        self._lock = threading.RLock()
        if load_models:
            self._load()
        else:
            self.errors.append(
                "Classifier loading was blocked because artifact integrity verification failed."
            )

    def _load(self) -> None:
        try:
            import joblib
        except ImportError:
            joblib = None

        bundle_candidates = sorted(self.models_directory.glob("*models.joblib"))
        if joblib and bundle_candidates:
            try:
                bundle = joblib.load(bundle_candidates[0])
                saved_names = list(bundle.get("class_names", []))
                if saved_names != self.config.class_names:
                    raise ValueError(f"Unexpected model class order: {saved_names}")
                self.models.update(bundle.get("models", {}))
                self.selected_model_name = str(
                    bundle.get("selected_model_name", self.selected_model_name)
                )
                self.bundle_metadata = {
                    "path": str(bundle_candidates[0]),
                    "feature_version": bundle.get("feature_version"),
                    "selected_model_name": self.selected_model_name,
                }
                if bundle.get("zoom_gap_calibration"):
                    self.config.zoom_gap_calibration = dict(bundle["zoom_gap_calibration"])
                if bundle.get("runtime", {}).get("confidence_floor") is not None:
                    self.config.confidence_floor = float(
                        bundle["runtime"]["confidence_floor"]
                    )
            except Exception as error:  # artifact diagnostics must not stop the dashboard
                self.errors.append(f"Could not load model bundle: {error}")

        online_state_path = self.models_directory / "gesture_online_state.joblib"
        if joblib and online_state_path.exists():
            try:
                online_state = joblib.load(online_state_path)
                if list(online_state.get("class_names", [])) != self.config.class_names:
                    raise ValueError("online checkpoint class order differs from v18_17")
                self.models["OnlineMLP"] = online_state["online_model"]
                self.selected_model_name = "OnlineMLP"
            except Exception as error:
                self.errors.append(f"Could not load online checkpoint: {error}")

        if {"SVM", "MLP"}.issubset(self.models):
            self.models["FollowObjectEnsemble"] = FollowObjectProbabilityEnsemble(
                self.models["SVM"], self.models["MLP"], len(self.config.class_names)
            )

        primary_path = self.models_directory / "gesture_primary_model.joblib"
        if joblib and primary_path.exists() and "Primary" not in self.models:
            try:
                self.models["Primary"] = joblib.load(primary_path)
            except Exception as error:
                self.errors.append(f"Could not load primary model: {error}")

        if self.selected_model_name not in self.models and self.models:
            self.selected_model_name = (
                "Primary" if "Primary" in self.models else next(iter(self.models))
            )
        selection_path = self.models_directory / "gesture_production_selection.json"
        if selection_path.exists():
            try:
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                if selection.get("pc_release_ready") is not True:
                    raise ValueError("selection did not pass the frozen PC quality gate")
                selected = str(selection.get("selected_model", ""))
                if selected not in self.models or selected == "FollowObjectEnsemble":
                    raise ValueError(f"qualified model is unavailable: {selected}")
                self.selected_model_name = selected
                self.bundle_metadata["production_selection"] = {
                    "path": str(selection_path),
                    "selected_model": selected,
                    "qualified_utc": selection.get("qualified_utc"),
                }
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                self.errors.append(f"Could not apply production model selection: {error}")

    @property
    def ready(self) -> bool:
        return bool(self.models)

    def status(self) -> dict[str, Any]:
        selectable = [
            name for name in self.models
            if name not in {"FollowObjectEnsemble", "OnlineMLPCandidate"}
        ]
        return {
            "ready": self.ready,
            "available_models": list(self.models),
            "selectable_models": selectable,
            "selected_model_name": self.selected_model_name,
            "follow_object_ensemble_ready": "FollowObjectEnsemble" in self.models,
            "errors": self.errors,
            "bundle": self.bundle_metadata,
        }

    def resolve_name(self, model_name: str | None = None) -> str:
        if not self.models:
            raise RuntimeError("No trained classifier artifact is available.")
        if model_name is not None and model_name not in self.models:
            raise ValueError(f"Unknown model selection: {model_name}")
        return model_name if model_name is not None else self.selected_model_name

    def activate_online_model(self, model: Any, *, select: bool = True) -> None:
        """Atomically activates an update or stages it behind the release gate."""
        with self._lock:
            if select:
                self.models["OnlineMLP"] = model
                self.selected_model_name = "OnlineMLP"
            elif self.selected_model_name == "OnlineMLP":
                # Keep the qualified OnlineMLP object live while the newly saved
                # checkpoint waits for a fresh release-qualification process.
                self.models["OnlineMLPCandidate"] = model
            else:
                self.models["OnlineMLP"] = model

    def predict(self, feature: np.ndarray, model_name: str | None = None) -> tuple[np.ndarray, str, float]:
        with self._lock:
            name = self.resolve_name(model_name)
            model = self.models[name]
            started = time.perf_counter()
            probabilities = np.asarray(
                model.predict_proba(np.asarray(feature, dtype=np.float32).reshape(1, -1))[0],
                dtype=np.float64,
            )
        classes = np.asarray(getattr(model, "classes_", np.arange(len(probabilities))), dtype=int)
        if not np.array_equal(classes, np.arange(len(self.config.class_names))):
            reordered = np.zeros(len(self.config.class_names), dtype=np.float64)
            reordered[classes] = probabilities
            probabilities = reordered
        probabilities /= max(float(probabilities.sum()), 1e-12)
        return probabilities, name, (time.perf_counter() - started) * 1000

    def predict_many(
        self, feature: np.ndarray, model_names: list[str]
    ) -> dict[str, tuple[np.ndarray, float]]:
        results: dict[str, tuple[np.ndarray, float]] = {}
        for name in dict.fromkeys(model_names):
            if name not in self.models:
                continue
            probabilities, _, elapsed_ms = self.predict(feature, name)
            results[name] = (probabilities, elapsed_ms)
        return results


class HandDetector:
    def __init__(self, model_path: Path):
        self.ready = False
        self.error: str | None = None
        self._landmarker = None
        self._mp = None
        if not model_path.exists():
            self.error = f"Missing MediaPipe hand model: {model_path.name}"
            return
        try:
            import mediapipe as mp

            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.35,
                min_hand_presence_confidence=0.35,
                min_tracking_confidence=0.5,
            )
            self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
            self._mp = mp
            self.ready = True
        except Exception as error:
            self.error = f"MediaPipe initialization failed: {error}"

    def detect(self, rgb_image: np.ndarray) -> np.ndarray | None:
        if not self.ready or self._landmarker is None or self._mp is None:
            return None
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_image, dtype=np.uint8),
        )
        result = self._landmarker.detect(image)
        candidates: list[tuple[float, np.ndarray]] = []
        for hand in result.hand_landmarks:
            points = np.asarray([[item.x, item.y] for item in hand], dtype=np.float32)
            if points.shape == (21, 2) and np.isfinite(points).all():
                area = float(np.ptp(points[:, 0]) * np.ptp(points[:, 1]))
                candidates.append((area, points))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def close(self) -> None:
        landmarker = self._landmarker
        self._landmarker = None
        self.ready = False
        if landmarker is not None and hasattr(landmarker, "close"):
            landmarker.close()


class InferenceEngine:
    def __init__(
        self,
        config: RuntimeConfig,
        models_directory: Path = MODELS_DIRECTORY,
        *,
        load_models: bool = True,
    ):
        self.config = config
        self.model_manager = ModelManager(
            config, models_directory, load_models=load_models
        )
        self.detector = HandDetector(models_directory / "hand_landmarker.task")
        self.resolver = GeometryResolver(config)
        self._lock = threading.Lock()

    def close(self) -> None:
        self.detector.close()

    def status(self) -> dict[str, Any]:
        return {
            "model": self.model_manager.status(),
            "mediapipe": {"ready": self.detector.ready, "error": self.detector.error},
            "ready": self.model_manager.ready and self.detector.ready,
        }

    def process_frame(
        self,
        image_bytes: bytes,
        session: RuntimeSession,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        now = time.monotonic()
        actual_fps = session.record_frame(now)
        try:
            import cv2
        except ImportError as error:
            return {"status": "dependency_missing", "message": str(error)}
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"status": "invalid_image", "message": "The frame could not be decoded."}
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.model_manager.ready:
            return {
                "status": "model_missing",
                "message": "Export the trained notebook artifacts into the models directory.",
                "actual_fps": actual_fps,
            }
        if not self.detector.ready:
            return {
                "status": "mediapipe_missing",
                "message": self.detector.error,
                "actual_fps": actual_fps,
            }

        with self._lock:
            landmark_started = time.perf_counter()
            landmarks = self.detector.detect(rgb)
            mediapipe_ms = (time.perf_counter() - landmark_started) * 1000
            if landmarks is None:
                session.temporal_gate.reset()
                session.diagnostic_gates.clear()
                follow = session.follow_object.observe(
                    "no_gesture", stable=False, confidence=0.0,
                    source="no hand", now=now,
                )
                total_ms = (time.perf_counter() - total_started) * 1000
                return {
                    "status": "no_hand",
                    "message": "No usable hand was found inside the guide.",
                    "runtime_prediction": "no_gesture",
                    "runtime_action": "Wait / No Action",
                    "confidence": 0.0,
                    "stable_frames": 0,
                    "follow_object": asdict(follow),
                    "actual_fps": actual_fps,
                    "timing": self._timing(
                        mediapipe_ms=mediapipe_ms,
                        feature_ms=0.0,
                        classifier_ms=0.0,
                        diagnostics_ms=0.0,
                        total_ms=total_ms,
                        effective_fps=actual_fps,
                    ),
                }
            feature_started = time.perf_counter()
            feature = landmarks_to_feature(landmarks)
            feature_ms = (time.perf_counter() - feature_started) * 1000
            follow_was_active = session.follow_object.active
            requested_name = (
                "FollowObjectEnsemble"
                if follow_was_active and "FollowObjectEnsemble" in self.model_manager.models
                else model_name
            )
            selected_name = self.model_manager.resolve_name(requested_name)
            diagnostic_names = [
                name for name in ("SVM", "MLP", "OnlineMLP", selected_name)
                if name in self.model_manager.models
            ]
            model_rows = self.model_manager.predict_many(feature, diagnostic_names)
            resolved_rows: dict[str, np.ndarray] = {}
            resolver_rows: dict[str, dict[str, dict | None]] = {}
            for name, (probabilities, _) in model_rows.items():
                resolved_rows[name], resolver_rows[name] = self.resolver.resolve(
                    probabilities, landmarks
                )

        raw_probabilities, classifier_ms = model_rows[selected_name]
        resolved_probabilities = resolved_rows[selected_name]
        resolver_details = resolver_rows[selected_name]
        decision = session.temporal_gate.update(resolved_probabilities, now=now)

        diagnostic_results: dict[str, dict[str, Any]] = {}
        for name, (probabilities, elapsed_ms) in model_rows.items():
            model_decision = (
                decision if name == selected_name
                else session.diagnostic_gate(name).update(resolved_rows[name], now=now)
            )
            diagnostic_results[name] = {
                "used_for_runtime": name == selected_name,
                "raw_prediction": self.config.class_names[int(np.argmax(probabilities))],
                "prediction": model_decision.predicted_gesture,
                "confidence": model_decision.confidence,
                "stable_frames": model_decision.stable_frames,
                "execute": model_decision.execute,
                "reason": model_decision.reason,
                "classifier_ms": elapsed_ms,
            }

        follow = session.follow_object.observe(
            decision.predicted_gesture,
            stable=bool(
                decision.execute
                and decision.confidence >= self.config.follow_min_confidence
            ),
            confidence=decision.confidence,
            source=selected_name,
            now=now,
        )
        mapped_action = self.config.gesture_to_action[decision.predicted_gesture]
        runtime_action = "Wait / No Action"
        action_reason = decision.reason
        if follow.completed:
            runtime_action = "Follow Object"
            action_reason = "palm → HaGRID fist → palm completed"
        elif follow_was_active:
            action_reason = follow.message
        elif decision.execute:
            new_action_edge = decision.predicted_gesture != session.last_action_gesture
            cooldown_ready = now - session.last_action_at >= self.config.action_cooldown_seconds
            if new_action_edge and cooldown_ready:
                runtime_action = mapped_action
                action_reason = "stable command emitted"
                session.last_action_gesture = decision.predicted_gesture
                session.last_action_at = now
            else:
                action_reason = "duplicate command suppressed"
        else:
            session.last_action_gesture = None

        total_ms = (time.perf_counter() - total_started) * 1000
        diagnostics_ms = float(sum(row[1] for row in model_rows.values()))
        return {
            "status": "predicted",
            "model": selected_name,
            "raw_prediction": self.config.class_names[int(np.argmax(raw_probabilities))],
            "runtime_prediction": decision.predicted_gesture,
            "mapped_action": mapped_action,
            "runtime_action": runtime_action,
            "action_reason": action_reason,
            "confidence": decision.confidence,
            "stable_frames": decision.stable_frames,
            "frames_used": decision.frames_used,
            "held_seconds": decision.held_seconds,
            "probabilities": decision.probabilities,
            "raw_probabilities": {
                name: float(raw_probabilities[index])
                for index, name in enumerate(self.config.class_names)
            },
            "follow_object": asdict(follow),
            "diagnostics": diagnostic_results,
            "resolvers": resolver_details,
            "landmarks": landmarks.round(6).tolist(),
            "feature_vector": feature.round(7).tolist(),
            "actual_fps": actual_fps,
            "timing": self._timing(
                mediapipe_ms=mediapipe_ms,
                feature_ms=feature_ms,
                classifier_ms=classifier_ms,
                diagnostics_ms=diagnostics_ms,
                total_ms=total_ms,
                effective_fps=actual_fps,
            ),
        }

    def _timing(
        self,
        *,
        mediapipe_ms: float,
        feature_ms: float,
        classifier_ms: float,
        diagnostics_ms: float,
        total_ms: float,
        effective_fps: float,
    ) -> dict[str, float | bool | str]:
        frame_budget = self.config.frame_budget_ms
        uncapped = 1000.0 / total_ms if total_ms > 0 else 0.0
        budget_used = 100.0 * total_ms / frame_budget
        capacity_pass = total_ms <= frame_budget
        return {
            "mediapipe_ms": float(mediapipe_ms),
            "feature_ms": float(feature_ms),
            "classifier_ms": float(classifier_ms),
            "diagnostics_ms": float(diagnostics_ms),
            "total_ms": float(total_ms),
            "uncapped_fps": float(uncapped),
            "configured_fps": float(self.config.target_fps),
            "effective_application_fps": float(
                min(self.config.target_fps, max(0.0, effective_fps))
            ),
            "frame_budget_ms": float(frame_budget),
            "budget_used_percent": float(budget_used),
            "headroom_ms": float(frame_budget - total_ms),
            "ten_fps_capacity_pass": bool(capacity_pass),
            "verdict": "PASS" if capacity_pass else "FAIL",
        }
