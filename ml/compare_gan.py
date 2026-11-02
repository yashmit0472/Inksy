"""Inference-only cGAN/VAE comparison: character proxy, diversity, grids and pages."""

import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from ml.characters import ALPHABET
from ml.checkpoint import load_checkpoint
from ml.gan import ConditionalGenerator
from ml.generate import sample_glyphs
from ml.model import ConditionalVAE
from ml.preview import DEMO_TEXT, create_preview
from ml.quality import diagnose
from ml.score_generation import score_generation
from ml.utils import sha256_file, write_json
from ml.visualize import save_grid


@torch.inference_mode()
def diversity_diagnostics(model, seed=2026, samples_per_class=20):
    """Screen exact repeats and low pixel variation; neither metric proves useful diversity."""
    if not 2 <= samples_per_class <= 100:
        raise ValueError("Use 2-100 samples per class for diversity diagnostics.")
    per_class = {}
    for label, char in enumerate(tqdm(ALPHABET, desc="Within-class variation", ascii=True)):
        images = sample_glyphs(model, char * samples_per_class, seed + label)
        mean_distance = float(torch.pdist(images.flatten(1), p=1).mean() / (28 * 28))
        quantized = (images.flatten(1) * 255).round().to(torch.uint8)
        per_class[char] = {
            "mean_pairwise_pixel_l1": mean_distance,
            "distinct_8bit_images": len(torch.unique(quantized, dim=0)),
        }
    return {
        "seed": seed,
        "samples_per_class": samples_per_class,
        "per_class": per_class,
        "mean_within_class_pixel_l1": sum(
            row["mean_pairwise_pixel_l1"] for row in per_class.values()
        )
        / len(ALPHABET),
        "warning": (
            "Exact uniqueness and pixel distance do not prove distinct handwriting styles. "
            "Noise can inflate variation; inspect repeated letters and failure cases."
        ),
    }


def summarize_comparison(baseline, candidate):
    def values(bundle):
        agreement = bundle["agreement"]["generated_agreement"]
        quality = bundle["quality"]["generated"]
        return {
            "exact_label_agreement": agreement["accuracy"],
            "case_insensitive_agreement": agreement["case_insensitive_accuracy"],
            "soft_ink_fraction": quality["soft_ink_fraction"],
            "mean_edge_energy": quality["mean_edge_energy"],
            "near_blank_fraction": quality["near_blank_fraction"],
            "mean_within_class_pixel_l1": bundle["diversity"]["mean_within_class_pixel_l1"],
        }

    before, after = values(baseline), values(candidate)
    return {
        "baseline": before,
        "candidate": after,
        "candidate_minus_baseline": {key: after[key] - before[key] for key in before},
        "vae_loss_comparison": None,
        "release_gate_passed": False,
        "human_review_completed": False,
        "training_performed": False,
        "test_set_evaluated": False,
        "warning": (
            "Different model families and training budgets: this is a practical candidate "
            "comparison, not an equal-compute ablation. The imperfect frozen recognizer is "
            "a proxy, not human accuracy. No automatic model selection or quality certification."
        ),
    }


def compare(
    candidate: Path,
    output: Path,
    *,
    baseline=Path("artifacts/refined-continue-5/best.pt"),
    recognizer=Path("artifacts/recognizer/best.pt"),
    root=Path("data"),
    prepared=Path("artifacts/data"),
):
    if output.exists():
        raise FileExistsError("Choose a new comparison output directory.")
    torch.set_num_threads(4)
    old_model, old_metadata = load_checkpoint(baseline)
    new_model, new_metadata = load_checkpoint(candidate)
    if not isinstance(old_model, ConditionalVAE) or not isinstance(new_model, ConditionalGenerator):
        raise ValueError("Expected our VAE baseline and a cGAN candidate checkpoint.")
    if old_metadata["split_sha256"] != new_metadata["split_sha256"]:
        raise ValueError("Models must use the same prepared data split.")
    output.mkdir(parents=True, exist_ok=False)
    bundles = {}
    for name, path, model in (
        ("baseline", baseline, old_model),
        ("candidate", candidate, new_model),
    ):
        folder = output / name
        agreement = score_generation(path, recognizer, folder / "agreement")
        quality = diagnose(path, folder / "quality", root=root, prepared=prepared)
        diversity = diversity_diagnostics(model)
        write_json(folder / "diversity.json", diversity)
        images = sample_glyphs(model, ALPHABET, seed=42, variations=4)
        names = [char for char in ALPHABET for _ in range(4)]
        save_grid(images, folder / "glyph-grid.png", names, columns=16)
        create_preview(DEMO_TEXT, checkpoint=path, output=folder / "paragraph", seed=42)
        bundles[name] = {"agreement": agreement, "quality": quality, "diversity": diversity}
    with (output / "human-review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "model",
                "grid_file",
                "grid_index_1based",
                "requested",
                "variation_1based",
                "observed_character",
                "readable_yes_no",
                "notes",
            ]
        )
        for model in ("baseline", "candidate"):
            for label, char in enumerate(ALPHABET):
                for variation in range(4):
                    writer.writerow(
                        [
                            model,
                            f"{model}/glyph-grid.png",
                            label * 4 + variation + 1,
                            char,
                            variation + 1,
                            "",
                            "",
                            "",
                        ]
                    )
    report = summarize_comparison(bundles["baseline"], bundles["candidate"])
    report.update(
        {
            "baseline_checkpoint": str(baseline),
            "candidate_checkpoint": str(candidate),
            "baseline_sha256": sha256_file(baseline),
            "candidate_sha256": sha256_file(candidate),
            "recognizer_sha256": sha256_file(recognizer),
            "split_sha256": old_metadata["split_sha256"],
            "seeds": {"agreement": 7301, "quality_and_diversity": 2026, "grids_and_pages": 42},
            "samples_per_class": {"agreement": 100, "quality_and_diversity": 20, "grids": 4},
            "quality_status": new_metadata.get("quality_status", "unreviewed"),
        }
    )
    write_json(output / "comparison.json", report)
    print(f"Saved {output / 'comparison.json'}. Human review is still required.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/comparison-cgan-pilot"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("artifacts/refined-continue-5/best.pt")
    )
    parser.add_argument("--recognizer", type=Path, default=Path("artifacts/recognizer/best.pt"))
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/data"))
    args = parser.parse_args()
    compare(
        args.checkpoint,
        args.output,
        baseline=args.baseline,
        recognizer=args.recognizer,
        root=args.root,
        prepared=args.prepared,
    )


if __name__ == "__main__":
    main()
