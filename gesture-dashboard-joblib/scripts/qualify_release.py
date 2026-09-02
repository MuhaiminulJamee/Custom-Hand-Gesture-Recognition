from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import load_production_settings, load_runtime_config
from backend.model_runtime import ModelManager
from backend.release_qualification import ReleaseQualifier


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the selected Python model on the untouched test cache. "
            "The no_gesture class is reported but deferred from this release gate."
        )
    )
    parser.add_argument("--models-directory", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    models_directory = arguments.models_directory.resolve()
    config = load_runtime_config(models_directory)
    settings = load_production_settings()
    manager = ModelManager(config, models_directory)
    qualifier = ReleaseQualifier(config, settings, models_directory)
    report = qualifier.run(manager)
    print(f"Report: {report['report']}")
    print(f"Selected model: {report['selected_model']}")
    print(f"Gated macro-F1: {report['gated_macro_f1']:.4f}")
    print(f"Minimum gated class F1: {report['gated_minimum_per_class_f1']:.4f}")
    print(f"Deferred classes: {', '.join(report['deferred_classes'])}")
    if report["pc_release_ready"]:
        print("PC release quality gate: PASS")
        return 0
    print(
        "PC release quality gate: FAIL | gated classes below threshold: "
        + ", ".join(report["failing_gated_classes"])
    )
    return 2 if arguments.require_pass else 0

if __name__ == "__main__":
    raise SystemExit(main())
