"""Validation diagnostics and human-review samples; no invented accuracy scores."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.characters import ALPHABET
from ml.checkpoint import load_checkpoint
from ml.dataset import load_emnist, load_partitions
from ml.generate import sample_glyphs
from ml.model import vae_loss
from ml.utils import resolve_device, seed_everything, sha256_file, write_json
from ml.visualize import save_grid


@torch.inference_mode()
def evaluate(
    checkpoint: Path,
    root: Path,
    prepared: Path,
    output: Path,
    *,
    device="cpu",
    partition="validation",
    final_test=False,
    max_batches=None,
    samples_per_class=20,
):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output must be a new or empty directory.")
    if partition == "test" and not final_test:
        raise ValueError("Final test data requires --final-test after model selection is frozen.")
    if partition not in {"validation", "test"}:
        raise ValueError("Partition must be validation or test.")
    if not 1 <= samples_per_class <= 100 or (max_batches is not None and max_batches < 1):
        raise ValueError("Use 1-100 samples/class and a positive batch limit.")
    seed_everything(43)
    torch.set_num_threads(4)
    device = resolve_device(device)
    model, metadata = load_checkpoint(checkpoint, device)
    _, _, validation, manifest = load_partitions(root, prepared)
    if metadata["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("Checkpoint and evaluation data use different splits.")
    dataset = load_emnist(root, train=False) if partition == "test" else validation
    loader = DataLoader(
        dataset, batch_size=256, shuffle=False, generator=torch.Generator().manual_seed(43)
    )
    sums = torch.zeros((len(ALPHABET), 3), dtype=torch.float64)
    counts = torch.zeros(len(ALPHABET), dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(43)
    beta = metadata["training_config"]["beta"]
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        mu, logvar = model.encode(images, labels)
        logits = model.decode(model.reparameterize(mu, logvar, generator), labels)
        for label in labels.unique():
            mask = labels == label
            losses = vae_loss(logits[mask], images[mask], mu[mask], logvar[mask], beta)
            index, count = int(label), int(mask.sum())
            sums[index] += (
                torch.tensor([losses[key].item() for key in ("loss", "reconstruction", "kl")])
                * count
            )
            counts[index] += count
    output.mkdir(parents=True, exist_ok=True)
    total = int(counts.sum())
    if total == 0:
        raise ValueError("Evaluation partition is empty.")
    report = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "quality_status": metadata["quality_status"],
        "partition": partition,
        "examples": total,
        "partial": max_batches is not None,
        "beta": beta,
        "metrics": dict(
            zip(("loss", "reconstruction", "kl"), (sums.sum(0) / total).tolist(), strict=True)
        ),
        "per_class": {
            char: {
                "count": int(counts[i]),
                "loss": float(sums[i, 0] / counts[i]) if counts[i] else None,
            }
            for i, char in enumerate(ALPHABET)
        },
        "class_accuracy": None,
        "warning": (
            "Reconstruction loss is not class accuracy. Human/classifier review is pending; "
            "this report does not certify release quality."
        ),
    }
    write_json(output / "evaluation.json", report)
    review_rows = []
    # Keep each class on its own sheet for requested-vs-observed human annotation.
    for index, char in enumerate(ALPHABET):
        images = sample_glyphs(model, char * samples_per_class, 1000 + index)
        filename = f"class-{index:02d}.png"
        save_grid(
            images,
            output / "review" / filename,
            [f"{char} #{i + 1}" for i in range(samples_per_class)],
            columns=10,
        )
        for sample in range(samples_per_class):
            review_rows.append([filename, sample + 1, char, "", "", ""])
    with (output / "human-review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sheet", "sample", "requested", "observed", "readable", "notes"])
        writer.writerows(review_rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "partition": partition,
                "examples": total,
                "class_accuracy": None,
            }
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--partition", choices=["validation", "test"], default="validation")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--samples-per-class", type=int, default=20)
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.root,
        args.prepared,
        args.output,
        device=args.device,
        partition=args.partition,
        final_test=args.final_test,
        max_batches=args.max_batches,
        samples_per_class=args.samples_per_class,
    )


if __name__ == "__main__":
    main()
