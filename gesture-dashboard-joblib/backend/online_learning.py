from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import secrets
import threading
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import f1_score

from .artifact_integrity import ArtifactRegistry
from .config import MODELS_DIRECTORY, RuntimeConfig
from .geometry import GeometryResolver


class OnlineLearningService:
    """Guarded, persistent OnlineMLP updates for reviewed PC feedback.

    The Windows dashboard updates only the trusted Python/Joblib checkpoint used
    by this application.
    """

    replay_per_class = 8
    history_per_class = 4
    new_sample_repeats = 8
    allowed_macro_f1_drop = 0.003
    allowed_per_class_f1_drop = 0.02

    def __init__(
        self,
        config: RuntimeConfig,
        model_manager: Any,
        models_directory: Path = MODELS_DIRECTORY,
        *,
        artifact_registry: ArtifactRegistry | None = None,
        force_learning_enabled: bool = False,
        promote_updates_live: bool = True,
    ):
        self.config = config
        self.model_manager = model_manager
        self.models_directory = Path(models_directory)
        self.state_path = self.models_directory / "gesture_online_state.joblib"
        self.replay_path = self.models_directory / "gesture_online_replay_cache.npz"
        self.validation_path = self.models_directory / "gesture_online_validation_cache.npz"
        self.examples_path = self.models_directory / "gesture_online_confirmed_features.npz"
        self.events_path = self.models_directory / "gesture_online_events_pc.csv"
        self.backup_directory = self.models_directory / "online_backups_pc"
        self.resolver = GeometryResolver(config)
        self.artifact_registry = artifact_registry
        self.force_learning_enabled = bool(force_learning_enabled)
        self.promote_updates_live = bool(promote_updates_live)
        self._lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        required = {
            "online_checkpoint": self.state_path.exists() or "OnlineMLP" in self.model_manager.models,
            "replay_cache": self.replay_path.exists(),
            "validation_cache": self.validation_path.exists(),
        }
        state: dict[str, Any] = {}
        if self.state_path.exists():
            try:
                state = joblib.load(self.state_path)
            except Exception:
                state = {}
        return {
            "ready": all(required.values()),
            "requirements": required,
            "accepted_updates": int(state.get("accepted_updates", 0)),
            "forced_updates": int(state.get("forced_updates", 0)),
            "rejected_updates": int(state.get("rejected_updates", 0)),
            "validation_macro_f1": state.get("validation_macro_f1"),
            "last_updated_utc": state.get("last_updated_utc"),
            "force_learning_enabled": self.force_learning_enabled,
            "updates_promoted_live": self.promote_updates_live,
            "backup_count": len(self.list_backups()),
            "checkpoint_policy": "validated Python/Joblib checkpoints only",
        }

    def list_backups(self) -> list[dict[str, Any]]:
        if not self.backup_directory.exists():
            return []
        rows = []
        for path in sorted(
            self.backup_directory.glob("state_before_*.joblib"), reverse=True
        ):
            try:
                rows.append({
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "modified_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                })
            except OSError:
                continue
        return rows

    def _refresh_manifest(self) -> None:
        if self.artifact_registry is not None:
            self.artifact_registry.build()

    def _prune_backups(self, keep: int = 50) -> None:
        backups = sorted(
            self.backup_directory.glob("state_before_*.joblib"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in backups[keep:]:
            path.unlink(missing_ok=True)

    @staticmethod
    def _prepare_mlp(model: Any) -> Any:
        if not hasattr(model, "named_steps") or not {"scaler", "mlp"}.issubset(model.named_steps):
            raise ValueError("Online learning requires the notebook's scaler + MLP pipeline.")
        classifier = model.named_steps["mlp"]
        classifier.early_stopping = False
        if getattr(classifier, "best_loss_", None) is None:
            finite = [
                float(value) for value in getattr(classifier, "loss_curve_", [])
                if value is not None and np.isfinite(value)
            ]
            classifier.best_loss_ = min(finite) if finite else float("inf")
        if getattr(classifier, "_no_improvement_count", None) is None:
            classifier._no_improvement_count = 0
        return model

    def _validation_metrics(self, model: Any, features: np.ndarray, labels: np.ndarray, landmarks: np.ndarray | None) -> dict[str, Any]:
        probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
        if landmarks is not None and len(landmarks) == len(features):
            probabilities = np.vstack([
                self.resolver.resolve(row, points)[0]
                for row, points in zip(probabilities, landmarks)
            ])
        predictions = probabilities.argmax(axis=1)
        per_class = f1_score(
            labels,
            predictions,
            labels=np.arange(len(self.config.class_names)),
            average=None,
            zero_division=0,
        )
        return {"macro_f1": float(per_class.mean()), "per_class_f1": per_class}

    @staticmethod
    def _atomic_joblib(value: Any, destination: Path) -> None:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        joblib.dump(value, temporary)
        temporary.replace(destination)

    @staticmethod
    def _atomic_npz(destination: Path, **arrays: np.ndarray) -> None:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)

    def _load_model_and_state(self, validation_x: np.ndarray, validation_y: np.ndarray, validation_landmarks: np.ndarray | None) -> tuple[Any, dict[str, Any]]:
        if self.state_path.exists():
            state = joblib.load(self.state_path)
            if list(state.get("class_names", [])) != self.config.class_names:
                raise ValueError("Online checkpoint class order does not match v18_17.")
            model = state["online_model"]
        else:
            base = self.model_manager.models.get("OnlineMLP") or self.model_manager.models.get("MLP")
            if base is None:
                raise FileNotFoundError("Neither OnlineMLP nor MLP is loaded.")
            model = deepcopy(base)
            metrics = self._validation_metrics(model, validation_x, validation_y, validation_landmarks)
            now = datetime.now(timezone.utc).isoformat()
            state = {
                "schema_version": 1,
                "online_model": model,
                "class_names": list(self.config.class_names),
                "feature_names": list(self.config.feature_names),
                "validation_macro_f1": metrics["macro_f1"],
                "validation_per_class_f1": metrics["per_class_f1"],
                "update_count": 0,
                "accepted_updates": 0,
                "forced_updates": 0,
                "rejected_updates": 0,
                "created_utc": now,
                "last_updated_utc": now,
            }
        return self._prepare_mlp(model), state

    def _append_example(self, feature: np.ndarray, label: int) -> tuple[np.ndarray, np.ndarray]:
        if self.examples_path.exists():
            with np.load(self.examples_path, allow_pickle=False) as cache:
                history_x = cache["X"].astype(np.float32)
                history_y = cache["y"].astype(np.int64)
        else:
            history_x = np.empty((0, len(self.config.feature_names)), dtype=np.float32)
            history_y = np.empty((0,), dtype=np.int64)
        history_x = np.vstack([history_x, feature]).astype(np.float32)
        history_y = np.concatenate([history_y, np.asarray([label], dtype=np.int64)])
        self._atomic_npz(self.examples_path, X=history_x, y=history_y)
        return history_x, history_y

    def _append_event(self, event: dict[str, Any]) -> None:
        fields = list(event)
        write_header = not self.events_path.exists()
        with self.events_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(event)

    def _synchronize_python_artifacts(self, model: Any, state: dict[str, Any]) -> None:
        self._atomic_joblib(state, self.state_path)
        self._atomic_joblib(model, self.models_directory / "gesture_primary_model.joblib")
        bundle_candidates = sorted(self.models_directory.glob("*models.joblib"))
        if bundle_candidates and self.promote_updates_live:
            bundle = joblib.load(bundle_candidates[0])
            bundle.setdefault("models", {})["OnlineMLP"] = model
            bundle["selected_model_name"] = "OnlineMLP"
            bundle["primary_model_name"] = "OnlineMLP"
            bundle["online_learning"] = {
                "enabled": True,
                "checkpoint": str(self.state_path),
                "update_count": int(state["update_count"]),
                "forced_updates": int(state.get("forced_updates", 0)),
                "validation_macro_f1": float(state["validation_macro_f1"]),
                "last_updated_utc": state["last_updated_utc"],
                "runtime": "Windows dashboard Python checkpoint",
            }
            self._atomic_joblib(bundle, bundle_candidates[0])
        self.model_manager.activate_online_model(model, select=self.promote_updates_live)

    def learn(self, feature_vector: list[float], actual_label: str, *, force: bool = False) -> dict[str, Any]:
        if force and not self.force_learning_enabled:
            raise PermissionError(
                "Force Learn is disabled in production. Use validation-gated Safe Learn."
            )
        if actual_label not in self.config.class_to_idx:
            raise ValueError(f"Unknown actual label: {actual_label}")
        feature = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        if feature.shape != (1, len(self.config.feature_names)) or not np.isfinite(feature).all():
            raise ValueError("A finite 76-D feature vector is required for online learning.")
        if not self.replay_path.exists() or not self.validation_path.exists():
            raise FileNotFoundError(
                "Copy gesture_online_replay_cache.npz and gesture_online_validation_cache.npz from Drive."
            )

        with self._lock:
            with np.load(self.replay_path, allow_pickle=False) as cache:
                replay_x = cache["X"].astype(np.float32)
                replay_y = cache["y"].astype(np.int64)
            with np.load(self.validation_path, allow_pickle=False) as cache:
                validation_x = cache["X"].astype(np.float32)
                validation_y = cache["y"].astype(np.int64)
                validation_landmarks = (
                    cache["landmarks"].astype(np.float32)
                    if "landmarks" in cache.files else None
                )
            expected_features = len(self.config.feature_names)
            if replay_x.ndim != 2 or replay_x.shape[1] != expected_features:
                raise ValueError("Replay cache does not contain exact 76-D features.")
            if validation_x.ndim != 2 or validation_x.shape[1] != expected_features:
                raise ValueError("Validation cache does not contain exact 76-D features.")
            if replay_y.shape != (len(replay_x),) or validation_y.shape != (len(validation_x),):
                raise ValueError("Online cache features and labels are not aligned.")
            if not np.isfinite(replay_x).all() or not np.isfinite(validation_x).all():
                raise ValueError("Online caches contain non-finite feature values.")
            model, state = self._load_model_and_state(
                validation_x, validation_y, validation_landmarks
            )
            label_index = self.config.class_to_idx[actual_label]
            history_x, history_y = self._append_example(feature, label_index)
            rng_seed = secrets.randbits(63)
            rng = np.random.default_rng(rng_seed)
            update_x = [np.repeat(feature, self.new_sample_repeats, axis=0)]
            update_y = [np.full(self.new_sample_repeats, label_index, dtype=np.int64)]
            for class_index in range(len(self.config.class_names)):
                available = np.flatnonzero(replay_y == class_index)
                if not len(available):
                    raise ValueError(f"Replay cache has no {self.config.class_names[class_index]} rows.")
                selected = rng.choice(
                    available,
                    size=self.replay_per_class,
                    replace=len(available) < self.replay_per_class,
                )
                update_x.append(replay_x[selected])
                update_y.append(replay_y[selected])
                history_available = np.flatnonzero(history_y[:-1] == class_index)
                if len(history_available):
                    history_selected = rng.choice(
                        history_available,
                        size=min(self.history_per_class, len(history_available)),
                        replace=False,
                    )
                    update_x.append(history_x[history_selected])
                    update_y.append(history_y[history_selected])

            batch_x = np.vstack(update_x).astype(np.float32)
            batch_y = np.concatenate(update_y)
            order = rng.permutation(len(batch_y))
            batch_x, batch_y = batch_x[order], batch_y[order]
            old_probability = float(model.predict_proba(feature)[0, label_index])
            candidate = self._prepare_mlp(deepcopy(model))
            transformed = candidate.named_steps["scaler"].transform(batch_x)
            candidate.named_steps["mlp"].partial_fit(transformed, batch_y)
            new_probability = float(candidate.predict_proba(feature)[0, label_index])
            candidate_metrics = self._validation_metrics(
                candidate, validation_x, validation_y, validation_landmarks
            )
            old_macro = float(state["validation_macro_f1"])
            old_per_class = np.asarray(state["validation_per_class_f1"], dtype=np.float64)
            macro_drop = old_macro - candidate_metrics["macro_f1"]
            class_drop = old_per_class - candidate_metrics["per_class_f1"]
            probability_gain = new_probability - old_probability
            safe = bool(
                macro_drop <= self.allowed_macro_f1_drop
                and float(class_drop.max()) <= self.allowed_per_class_f1_drop
                and probability_gain > 0.0
            )
            accepted = bool(safe or force)
            now = datetime.now(timezone.utc)
            if accepted:
                self.backup_directory.mkdir(parents=True, exist_ok=True)
                backup = self.backup_directory / f"state_before_{int(state.get('update_count', 0)) + 1:06d}_{now.strftime('%Y%m%dT%H%M%SZ')}.joblib"
                self._atomic_joblib(state, backup)
                state["online_model"] = candidate
                state["validation_macro_f1"] = candidate_metrics["macro_f1"]
                state["validation_per_class_f1"] = candidate_metrics["per_class_f1"]
                state["accepted_updates"] = int(state.get("accepted_updates", 0)) + 1
                state["update_count"] = int(state.get("update_count", 0)) + 1
                if force and not safe:
                    state["forced_updates"] = int(state.get("forced_updates", 0)) + 1
                state["last_updated_utc"] = now.isoformat()
                self._synchronize_python_artifacts(candidate, state)
                self._prune_backups()
            else:
                state["rejected_updates"] = int(state.get("rejected_updates", 0)) + 1
                state["last_updated_utc"] = now.isoformat()
                self._atomic_joblib(state, self.state_path)

            event = {
                "timestamp_utc": now.isoformat(),
                "actual_gesture": actual_label,
                "accepted": accepted,
                "force_requested": force,
                "safety_gate_passed": safe,
                "probability_before": old_probability,
                "probability_after": new_probability,
                "probability_gain": probability_gain,
                "validation_macro_f1_before": old_macro,
                "validation_macro_f1_after": candidate_metrics["macro_f1"],
                "maximum_per_class_f1_drop": float(class_drop.max()),
                "joblib_checkpoint_updated": accepted,
            }
            self._append_event(event)
            self._refresh_manifest()
            if not accepted:
                message = "Update rejected by the validation safety gate; the live model is unchanged."
            elif self.promote_updates_live:
                message = "Python OnlineMLP checkpoint updated and is live."
            else:
                message = (
                    "Python OnlineMLP candidate updated. The qualified production model "
                    "remains live until the candidate passes release qualification."
                )
            return {
                "status": "accepted" if accepted else "rejected",
                **event,
                "rng_seed": rng_seed,
                "message": message,
            }

    def rollback(self, backup_name: str) -> dict[str, Any]:
        if not backup_name or Path(backup_name).name != backup_name:
            raise ValueError("A valid backup file name is required.")
        backup_path = (self.backup_directory / backup_name).resolve()
        if backup_path.parent != self.backup_directory.resolve():
            raise ValueError("Backup path is outside the managed backup directory.")
        if not backup_path.is_file() or not backup_path.name.startswith("state_before_"):
            raise FileNotFoundError("The requested online-learning backup does not exist.")
        with self._lock:
            restored = joblib.load(backup_path)
            if list(restored.get("class_names", [])) != self.config.class_names:
                raise ValueError("Backup class order does not match the runtime contract.")
            model = self._prepare_mlp(restored["online_model"])
            current = joblib.load(self.state_path) if self.state_path.exists() else None
            now = datetime.now(timezone.utc)
            if current is not None:
                self.backup_directory.mkdir(parents=True, exist_ok=True)
                rollback_backup = self.backup_directory / (
                    f"state_before_rollback_{now.strftime('%Y%m%dT%H%M%SZ')}.joblib"
                )
                self._atomic_joblib(current, rollback_backup)
            restored["online_model"] = model
            restored["last_updated_utc"] = now.isoformat()
            self._synchronize_python_artifacts(model, restored)
            self._prune_backups()
            self._refresh_manifest()
            return {
                "status": "rolled_back",
                "backup": backup_name,
                "validation_macro_f1": restored.get("validation_macro_f1"),
                "update_count": restored.get("update_count", 0),
                "last_updated_utc": restored["last_updated_utc"],
            }
