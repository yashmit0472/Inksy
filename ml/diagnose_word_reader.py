"""Inference-only image ablations for a trained content reader."""

import argparse
from collections import Counter
from pathlib import Path

import torch

from ml.train_word_reader import evaluation_loader
from ml.utils import sha256_file, write_json
from ml.word_reader import greedy_decode, load_reader, reading_metrics, word_ctc_loss


def ablate_images(images, widths, condition):
    """Preserve canvas and true widths so sequence length cannot explain differences."""
    if condition == "real":
        return images.clone()
    if condition == "blank":
        return torch.zeros_like(images)
    if condition != "reversed":
        raise ValueError("Unknown image ablation.")
    result = torch.zeros_like(images)
    for index, width in enumerate(widths.tolist()):
        result[index, :, :, :width] = images[index, :, :, :width].flip(-1)
    return result


@torch.inference_mode()
def diagnose(model, loader):
    model.eval()
    conditions = ("real", "blank", "reversed")
    rows = {name: [] for name in conditions}
    losses = dict.fromkeys(conditions, 0.0)
    changes = dict.fromkeys(conditions[1:], 0)
    distances = dict.fromkeys(conditions[1:], 0.0)
    count, frames = 0, 0
    for batch in loader:
        reference, lengths = model(batch["images"], batch["widths"])
        actual = greedy_decode(reference, lengths)
        valid = torch.arange(len(reference))[:, None] < lengths[None, :]
        frames += int(valid.sum())
        count += len(actual)
        for name in conditions:
            probabilities = reference
            if name != "real":
                probabilities, _ = model(
                    ablate_images(batch["images"], batch["widths"], name), batch["widths"]
                )
            predicted = greedy_decode(probabilities, lengths)
            losses[name] += float(word_ctc_loss(probabilities, lengths, batch)) * len(actual)
            rows[name].extend(
                {"id": identifier, "expected": text, "predicted": prediction}
                for identifier, text, prediction in zip(
                    batch["ids"], batch["text"], predicted, strict=True
                )
            )
            if name != "real":
                changes[name] += sum(a != b for a, b in zip(actual, predicted, strict=True))
                tv = (reference.exp() - probabilities.exp()).abs().sum(-1) / 2
                distances[name] += float(tv[valid].sum())
    if not count:
        raise ValueError("Diagnostics require at least one example.")
    metrics = {}
    for name in conditions:
        metrics[name] = {
            **reading_metrics(rows[name]),
            "ctc_loss": losses[name] / count,
            "prediction_counts": dict(Counter(row["predicted"] for row in rows[name])),
        }
        if name != "real":
            metrics[name]["changed_prediction_fraction"] = changes[name] / count
            metrics[name]["mean_frame_probability_total_variation"] = distances[name] / frames
    return metrics, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/iam-words"))
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new diagnostic output directory.")
    torch.set_num_threads(4)
    checksum = sha256_file(args.checkpoint)
    model, metadata = load_reader(args.checkpoint)
    model.requires_grad_(False)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    loader, identity = evaluation_loader(args.prepared, "content_validation", 16, args.max_batches)
    if metadata["records_sha256"] != identity["records_sha256"]:
        raise ValueError("Checkpoint and data identities differ.")
    metrics, rows = diagnose(model, loader)
    unchanged = all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
    if not unchanged or sha256_file(args.checkpoint) != checksum:
        raise RuntimeError("Reader state changed during diagnostics.")
    write_json(
        args.output / "diagnostics.json",
        {
            "checkpoint_sha256": checksum,
            "source_sha256": sha256_file(Path(__file__)),
            "partition": identity,
            "conditions": metrics,
            "predictions": rows,
            "training_performed": False,
            "weights_unchanged": unchanged,
            "test_set_evaluated": False,
            "interpretation_limit": (
                "Image sensitivity is not reading competence or generator quality."
            ),
        },
    )
    print(metrics)


if __name__ == "__main__":
    main()
