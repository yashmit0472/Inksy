"""Inference-only paired generator comparison; never initiates model training."""

import argparse
from pathlib import Path

from ml.evaluate import evaluate
from ml.quality import diagnose
from ml.score_generation import score_generation
from ml.utils import sha256_file, write_json


def summarize_comparison(baseline: dict, candidate: dict) -> dict:
    """Common raw metrics and deltas; no automatic claim that an experiment passed quality gates."""

    def row(bundle):
        agreement = bundle["agreement"]["generated_agreement"]
        metrics = bundle["validation"]["metrics"]
        generated = bundle["quality"]["generated"]
        return {
            "common_validation_loss": metrics["loss"],
            "validation_edge_error": metrics["edge"],
            "exact_label_agreement": agreement["accuracy"],
            "case_insensitive_agreement": agreement["case_insensitive_accuracy"],
            "soft_ink_fraction": generated["soft_ink_fraction"],
            "edge_energy": generated["mean_edge_energy"],
            "near_blank_fraction": generated["near_blank_fraction"],
        }

    original, updated = row(baseline), row(candidate)
    return {
        "baseline": original,
        "candidate": updated,
        "candidate_minus_baseline": {key: updated[key] - original[key] for key in original},
        "release_gate_passed": False,
        "interpretation": (
            "Lower common validation loss/edge error and higher agreement are "
            "diagnostic directions. "
            "Edge statistics depend on shapes; more edge energy can also be noise. "
            "Review fixed-seed grids, errors, case ambiguity and variation "
            "before accepting a change. "
            "The frozen recognizer is imperfect; agreement is not human accuracy."
        ),
    }


def compare_generators(
    baseline_checkpoint: Path,
    candidate_checkpoint: Path,
    recognizer: Path,
    output: Path,
    *,
    root=Path("data"),
    prepared=Path("artifacts/data"),
):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Comparison output must be new or empty.")
    # Fail before work or artifact creation if an expected checkpoint is missing.
    hashes = {
        "baseline": sha256_file(baseline_checkpoint),
        "candidate": sha256_file(candidate_checkpoint),
        "recognizer": sha256_file(recognizer),
    }
    if hashes["baseline"] == hashes["candidate"]:
        raise ValueError("Baseline and candidate are the same checkpoint.")
    output.mkdir(parents=True, exist_ok=True)
    bundles = {}
    for name, checkpoint in (
        ("baseline", baseline_checkpoint),
        ("candidate", candidate_checkpoint),
    ):
        print(
            f"Evaluating {name} with fixed validation data and paired sample seeds...", flush=True
        )
        validation = evaluate(checkpoint, root, prepared, output / name / "validation")
        agreement = score_generation(
            checkpoint, recognizer, output / name / "agreement", samples_per_class=100, seed=7301
        )
        quality = diagnose(
            checkpoint,
            output / name / "quality",
            root=root,
            prepared=prepared,
            samples_per_class=20,
            seed=2026,
        )
        bundles[name] = {"validation": validation, "agreement": agreement, "quality": quality}
    audit = bundles["baseline"]["agreement"]["recognizer_real_audit"]
    report = {
        "checkpoint_sha256": hashes,
        "test_set_evaluated": False,
        "paired_seeds": {"validation_noise": 43, "generation": 7301, "quality": 2026},
        "common_validation_objective": {"beta": 1.0, "edge_weight": 0.0},
        "recognizer_real_audit": {
            key: audit[key]
            for key in ("examples", "accuracy", "macro_accuracy", "case_insensitive_accuracy")
        },
        **summarize_comparison(bundles["baseline"], bundles["candidate"]),
    }
    write_json(output / "comparison.json", report)
    print(f"Comparison saved to {output / 'comparison.json'}. Visual review is still required.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/baseline-10/best.pt"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, default=Path("artifacts/recognizer/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/comparison-refined"))
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/data"))
    args = parser.parse_args()
    compare_generators(
        args.baseline,
        args.candidate,
        args.recognizer,
        args.output,
        root=args.root,
        prepared=args.prepared,
    )


if __name__ == "__main__":
    main()
