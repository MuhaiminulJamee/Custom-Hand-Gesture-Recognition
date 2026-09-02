from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from .artifact_integrity import sha256_file
from .config import MODELS_DIRECTORY, ProductionSettings, RuntimeConfig
from .geometry import GeometryResolver


QUALIFICATION_NAME = "gesture_production_qualification.json"
SELECTION_NAME = "gesture_production_selection.json"
PRODUCTION_CANDIDATES = ("SVM", "MLP", "OnlineMLP")
SELECTION_METRIC_TOLERANCE = 0.002
DEPLOYMENT_PREFERENCE = {"MLP": 3, "OnlineMLP": 2, "SVM": 1}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class ReleaseQualifier:
    """Selects a frozen Python runtime using the untouched cache.

    ``no_gesture`` is still measured and reported, but is intentionally excluded
    from this gate until its dedicated dataset improvement is completed.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        settings: ProductionSettings,
        models_directory: Path = MODELS_DIRECTORY,
    ):
        self.config = config
        self.settings = settings
        self.models_directory = Path(models_directory)
        self.output_path = self.models_directory / QUALIFICATION_NAME
        self.selection_path = self.models_directory / SELECTION_NAME
        self.validation_path = self.models_directory / "gesture_online_validation_cache.npz"
        self.test_path = self.models_directory / "gesture_online_untouched_test_cache.npz"

    def _model_source(self, model_name: str) -> Path | None:
        if model_name == "OnlineMLP":
            candidate = self.models_directory / "gesture_online_state.joblib"
            if candidate.exists():
                return candidate
        if model_name in {"SVM", "MLP"}:
            bundles = sorted(self.models_directory.glob("*models.joblib"))
            if bundles:
                return bundles[0]
        primary = self.models_directory / "gesture_primary_model.joblib"
        if primary.exists():
            return primary
        return None

    def _source_fingerprints(self, model_name: str) -> dict[str, str] | None:
        model_source = self._model_source(model_name)
        if (
            model_source is None
            or not self.validation_path.exists()
            or not self.test_path.exists()
        ):
            return None
        return {
            "model_file": model_source.name,
            "model_sha256": sha256_file(model_source),
            "test_cache_file": self.test_path.name,
            "test_cache_sha256": sha256_file(self.test_path),
            "validation_cache_file": self.validation_path.name,
            "validation_cache_sha256": sha256_file(self.validation_path),
        }

    def status(self, selected_model: str) -> dict[str, Any]:
        if not self.output_path.exists():
            return {
                "status": "not_run",
                "pc_release_ready": False,
                "report": str(self.output_path),
                "message": "Run scripts\\qualify_release.py to create a frozen release report.",
            }
        try:
            report = json.loads(self.output_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {
                "status": "invalid_report",
                "pc_release_ready": False,
                "report": str(self.output_path),
            }
        current = self._source_fingerprints(selected_model)
        report["current"] = bool(
            selected_model == report.get("selected_model")
            and current
            and report.get("sources") == current
        )
        if not report["current"]:
            report["status"] = "stale"
            report["pc_release_ready"] = False
        return report

    def _evaluate_model(
        self,
        model_name: str,
        model: Any,
        features: np.ndarray,
        labels: np.ndarray,
        landmarks: np.ndarray,
    ) -> dict[str, Any]:
        probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
        model_classes = np.asarray(
            getattr(model, "classes_", np.arange(probabilities.shape[1])), dtype=int
        )
        if not np.array_equal(model_classes, np.arange(len(self.config.class_names))):
            aligned = np.zeros((len(features), len(self.config.class_names)), dtype=np.float64)
            aligned[:, model_classes] = probabilities
            probabilities = aligned
        resolver = GeometryResolver(self.config)
        resolved = np.vstack([
            resolver.resolve(row, points)[0]
            for row, points in zip(probabilities, landmarks)
        ])
        predictions = resolved.argmax(axis=1)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            labels=np.arange(len(self.config.class_names)),
            zero_division=0,
        )
        class_metrics = {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "release_gate_included": name not in self.settings.deferred_quality_classes,
            }
            for index, name in enumerate(self.config.class_names)
        }
        included_indexes = [
            index for index, name in enumerate(self.config.class_names)
            if name not in self.settings.deferred_quality_classes
        ]
        gated_macro = float(np.mean(f1[included_indexes]))
        gated_minimum = float(np.min(f1[included_indexes]))
        failing = [
            self.config.class_names[index]
            for index in included_indexes
            if f1[index] < self.settings.minimum_per_class_f1
        ]
        passed = bool(gated_macro >= self.settings.minimum_macro_f1 and not failing)
        return {
            "model": model_name,
            "passed": passed,
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "gated_macro_f1": gated_macro,
            "gated_minimum_per_class_f1": gated_minimum,
            "failing_gated_classes": failing,
            "class_metrics": class_metrics,
        }

    def run(self, model_manager: Any) -> dict[str, Any]:
        if not self.validation_path.exists() or not self.test_path.exists():
            raise FileNotFoundError("Validation and untouched test caches are required.")
        with np.load(self.validation_path, allow_pickle=False) as cache:
            validation_features = cache["X"].astype(np.float32)
            validation_labels = cache["y"].astype(np.int64)
            validation_landmarks = cache["landmarks"].astype(np.float32)
        with np.load(self.test_path, allow_pickle=False) as cache:
            features = cache["X"].astype(np.float32)
            labels = cache["y"].astype(np.int64)
            landmarks = cache["landmarks"].astype(np.float32)
        expected_features = len(self.config.feature_names)
        for split_name, split_x, split_y, split_landmarks in (
            ("Validation", validation_features, validation_labels, validation_landmarks),
            ("Untouched test", features, labels, landmarks),
        ):
            if split_x.ndim != 2 or split_x.shape[1] != expected_features:
                raise ValueError(f"{split_name} cache does not contain exact 76-D features.")
            if split_y.shape != (len(split_x),) or split_landmarks.shape != (len(split_x), 21, 2):
                raise ValueError(f"{split_name} cache arrays are not aligned.")
            if not np.isfinite(split_x).all() or not np.isfinite(split_landmarks).all():
                raise ValueError(f"{split_name} cache contains non-finite values.")

        candidate_names = [
            name for name in PRODUCTION_CANDIDATES if name in model_manager.models
        ]
        if not candidate_names:
            candidate_names = [model_manager.selected_model_name]
        results = [
            self._evaluate_model(
                name,
                model_manager.models[name],
                validation_features,
                validation_labels,
                validation_landmarks,
            )
            for name in candidate_names
        ]
        passing = [result for result in results if result["passed"]]
        ranked = passing or results
        best_validation_macro = max(result["gated_macro_f1"] for result in ranked)
        statistically_tied = [
            result for result in ranked
            if best_validation_macro - result["gated_macro_f1"] <= SELECTION_METRIC_TOLERANCE
        ]
        validation_selected = max(
            statistically_tied,
            key=lambda result: DEPLOYMENT_PREFERENCE.get(result["model"], 0),
        )
        selected = self._evaluate_model(
            validation_selected["model"],
            model_manager.models[validation_selected["model"]],
            features,
            labels,
            landmarks,
        )
        pc_ready = bool(validation_selected["passed"] and selected["passed"])
        now = datetime.now(timezone.utc).isoformat()
        sources = self._source_fingerprints(selected["model"])
        selection = {
            "schema_version": 1,
            "qualified_utc": now,
            "selected_model": selected["model"],
            "pc_release_ready": pc_ready,
            "sources": sources,
            "thresholds": {
                "minimum_macro_f1": self.settings.minimum_macro_f1,
                "minimum_per_class_f1": self.settings.minimum_per_class_f1,
            },
            "deferred_classes": list(self.settings.deferred_quality_classes),
            "selection_basis": {
                "split": "validation",
                "metric": "gated_macro_f1",
                "near_tie_tolerance": SELECTION_METRIC_TOLERANCE,
                "near_tie_preference": ["MLP", "OnlineMLP", "SVM"],
                "untouched_test_used_for_selection": False,
            },
        }
        _atomic_json(self.selection_path, selection)
        report = {
            "schema_version": 1,
            "status": "passed" if pc_ready else "failed",
            "created_utc": now,
            "current": True,
            "pc_release_ready": pc_ready,
            "selected_model": selected["model"],
            "samples": len(features),
            "validation_samples": len(validation_features),
            "accuracy": selected["accuracy"],
            "balanced_accuracy": selected["balanced_accuracy"],
            "gated_macro_f1": selected["gated_macro_f1"],
            "gated_minimum_per_class_f1": selected["gated_minimum_per_class_f1"],
            "thresholds": selection["thresholds"],
            "deferred_classes": selection["deferred_classes"],
            "failing_gated_classes": selected["failing_gated_classes"],
            "class_metrics": selected["class_metrics"],
            "candidate_results": {
                result["model"]: {
                    key: value for key, value in result.items()
                    if key != "class_metrics"
                }
                for result in results
            },
            "selection_validation_result": {
                key: value for key, value in validation_selected.items()
                if key != "class_metrics"
            },
            "runtime_format": "Joblib",
            "sources": sources,
            "selection": str(self.selection_path),
            "report": str(self.output_path),
        }
        _atomic_json(self.output_path, report)
        return report
