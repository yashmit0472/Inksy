"""Descriptive glyph diagnostics, not recognition or perceptual quality scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ml.characters import ALPHABET
from ml.checkpoint import load_checkpoint
from ml.dataset import load_partitions
from ml.generate import sample_glyphs
from ml.utils import sha256_file, write_json
from ml.visualize import save_grid


def glyph_statistics(images: torch.Tensor) -> dict:
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28) or len(images) == 0:
        raise ValueError("Expected nonempty N-by-1-by-28-by-28 glyphs.")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Glyph intensities must be finite and in [0, 1].")
    active = images > 0.1
    soft = active & (images < 0.9)
    soft_fraction = soft.flatten(1).sum(1) / active.flatten(1).sum(1).clamp_min(1)
    dx = images[:, :, :, 1:] - images[:, :, :, :-1]
    dy = images[:, :, 1:, :] - images[:, :, :-1, :]
    energy = 0.5 * (dx.square().flatten(1).mean(1) + dy.square().flatten(1).mean(1))
    return {
        "images": len(images),
        "mean_ink_intensity": float(images.mean()),
        "soft_ink_fraction": float(soft_fraction.mean()),
        "mean_edge_energy": float(energy.mean()),
        "near_blank_fraction": float((images.flatten(1).amax(1) < 0.3).float().mean()),
    }


@torch.inference_mode()
def diagnose(
    checkpoint: Path,
    output: Path,
    *,
    root=Path("data"),
    prepared=Path("artifacts/data"),
    samples_per_class=20,
    seed=2026,
):
    if not 2 <= samples_per_class <= 100:
        raise ValueError("Use 2-100 samples per class.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Diagnostics require a new or empty output directory.")
    torch.set_num_threads(4)
    model, metadata = load_checkpoint(checkpoint, "cpu")
    dataset, _, validation, manifest = load_partitions(root, prepared)
    if metadata["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("Checkpoint and reference data have different partitions.")
    targets = dataset.targets[validation.indices].numpy()
    reference_indices = np.asarray(validation.indices)
    rng = np.random.default_rng(seed)
    per_class, real_all, generated_all = {}, [], []
    output.mkdir(parents=True, exist_ok=True)
    for label, char in enumerate(ALPHABET):
        candidates = reference_indices[targets == label]
        if len(candidates) < samples_per_class:
            raise ValueError(f"Not enough held-out references for class {char}.")
        selected = rng.choice(candidates, samples_per_class, replace=False)
        real = torch.stack([dataset[int(index)][0] for index in selected])
        generated = sample_glyphs(model, char * samples_per_class, seed + label)
        per_class[char] = {"real": glyph_statistics(real), "generated": glyph_statistics(generated)}
        real_all.append(real)
        generated_all.append(generated)
        # Representative, fixed-seed pairs. Both images are 28x28 internally; no enhancement.
        pairs = torch.stack(
            [image for pair in zip(real[:4], generated[:4], strict=True) for image in pair]
        )
        names = [f"{char} {kind}" for _ in range(4) for kind in ("real", "gen")]
        save_grid(pairs, output / "comparisons" / f"class-{label:02d}.png", names, columns=8)
    report = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "seed": seed,
        "samples_per_class": samples_per_class,
        "total_per_domain": samples_per_class * len(ALPHABET),
        "real_reference": glyph_statistics(torch.cat(real_all)),
        "generated": glyph_statistics(torch.cat(generated_all)),
        "per_class": per_class,
        "class_accuracy": None,
        "test_set_evaluated": False,
        "metric_notes": {
            "soft_ink_fraction": (
                "Among pixels above 0.1 ink intensity, fraction below 0.9. Higher values indicate "
                "more intermediate intensities, not necessarily worse readability."
            ),
            "mean_edge_energy": (
                "Mean squared horizontal/vertical adjacent-pixel differences. Sensitive to shape "
                "and stroke coverage as well as blur; not a perceptual quality score."
            ),
            "reference": (
                "Balanced sample of real validation examples. Generated samples use independent "
                "prior noise, not an encoded reference."
            ),
        },
    }
    write_json(output / "diagnostics.json", report)
    print(
        json.dumps(
            {"real_reference": report["real_reference"], "generated": report["generated"]}, indent=2
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/quality-baseline-10"))
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    diagnose(
        args.checkpoint,
        args.output,
        root=args.root,
        prepared=args.prepared,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
