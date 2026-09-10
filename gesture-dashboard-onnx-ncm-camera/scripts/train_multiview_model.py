"""Train and qualify a multi-view directional ONNX candidate.

This command never overwrites the production model.  It writes a candidate and
evidence under ``artifacts/multiview``.  Coverage must pass unless the explicitly
exploratory ``--allow-incomplete-coverage`` flag is supplied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend.config import load_runtime_config
from research.multiview import (
    capture_subset,
    coverage_audit,
    load_capture_manifest,
    make_balanced_training_set,
    qualify_candidate,
    train_candidate,
    write_history,
    write_json,
)
from research.v18_20 import load_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a leakage-safe ONNX candidate from reviewed multi-view captures."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/multiview"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/multiview"))
    parser.add_argument("--per-class", type=int, default=4096)
    parser.add_argument("--capture-share", type=float, default=0.25)
    parser.add_argument("--maximum-capture-repeats", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1900)
    parser.add_argument(
        "--allow-incomplete-coverage",
        action="store_true",
        help="Build an exploratory candidate even when participant coverage is insufficient.",
    )
    parser.add_argument(
        "--audit-only", action="store_true", help="Validate the manifest and report coverage only."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    dataset = (project / args.dataset).resolve() if not args.dataset.is_absolute() else args.dataset.resolve()
    output = (project / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if (dataset / "dataset.json").is_file():
        from research.collection_dataset import load_collection_dataset
        captured, ingestion = load_collection_dataset(dataset)
        write_json(output / "collection_ingestion.json", ingestion)
        print(json.dumps(ingestion, indent=2))
    else:
        captured = load_capture_manifest(dataset)
    coverage = coverage_audit(captured)
    write_json(output / "coverage_audit.json", coverage)
    print(json.dumps(coverage, indent=2))
    if args.audit_only:
        return
    if not coverage["passed"] and not args.allow_incomplete_coverage:
        detail = "\n".join(f"- {issue}" for issue in coverage["issues"])
        raise SystemExit(
            "Multi-view coverage is not sufficient for a qualified model.\n"
            f"{detail}\nCollect more participants or use --allow-incomplete-coverage "
            "for an exploratory, non-release candidate."
        )

    captured_train = capture_subset(captured, "train")
    captured_validation = capture_subset(captured, "val")
    captured_test = capture_subset(captured, "test")
    base_training = load_npz(project / "artifacts/v18_20/balanced_train.npz")
    public_validation = load_npz(project / "artifacts/v18_20/public_val.npz")
    public_test = load_npz(project / "artifacts/v18_20/public_test.npz")
    training, balance = make_balanced_training_set(
        base_training,
        captured_train,
        per_class=args.per_class,
        capture_share=args.capture_share,
        maximum_capture_repeats=args.maximum_capture_repeats,
        seed=args.seed,
    )
    write_json(output / "training_balance.json", balance)
    with (output / "balanced_train.npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, **training)
    (output / "balanced_train.npz.tmp").replace(output / "balanced_train.npz")

    candidate = output / "candidate.onnx"
    history = train_candidate(
        project,
        training,
        public_validation,
        captured_validation,
        candidate,
        epochs=args.epochs,
        seed=args.seed,
    )
    write_history(output / "training_history.csv", history)
    config = load_runtime_config(project / "models")
    qualification = qualify_candidate(
        project / "models/gesture_mlp_production.onnx",
        candidate,
        public_test,
        captured_test,
        config,
        coverage,
    )
    qualification["exploratory"] = bool(not coverage["passed"])
    qualification["production_model_changed"] = False
    qualification["training"] = balance
    write_json(output / "qualification.json", qualification)
    print(json.dumps(qualification, indent=2))
    if qualification["passed"]:
        print(
            "Candidate passed offline gates. The production model was not changed; "
            "complete the live NCM action test before promotion."
        )
    else:
        print(
            "Candidate did not pass release gates. The production model was not changed."
        )


if __name__ == "__main__":
    main()
