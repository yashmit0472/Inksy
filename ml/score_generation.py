"""Inference-only requested-character agreement against an audited, from-scratch recognizer."""

import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from ml.characters import ALPHABET
from ml.checkpoint import load_checkpoint
from ml.generate import sample_glyphs
from ml.recognizer import confusion_metrics, load_recognizer
from ml.utils import resolve_device, sha256_file, write_json


@torch.inference_mode()
def score_generation(
    generator_checkpoint: Path,
    recognizer_checkpoint: Path,
    output: Path,
    *,
    samples_per_class=100,
    seed=7301,
    device="cpu",
):
    if not 20 <= samples_per_class <= 1000:
        raise ValueError("Use 20-1000 generated samples per class.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new or empty score output directory.")
    torch.set_num_threads(4)
    device = resolve_device(device)
    generator, generator_metadata = load_checkpoint(generator_checkpoint, device)
    recognizer, recognizer_metadata = load_recognizer(recognizer_checkpoint, device)
    if recognizer_metadata.get("quality_status") != "audited-research-scorer":
        raise ValueError(
            "The recognizer needs a full owner-run training and real-data audit first."
        )
    if generator_metadata["split_sha256"] != recognizer_metadata["split_sha256"]:
        raise ValueError("Generator and recognizer use different source-data partitions.")
    audit = recognizer_metadata["real_audit"]
    if audit["classes_evaluated"] != len(ALPHABET):
        raise ValueError("The recognizer audit does not cover all supported classes.")
    confusion = torch.zeros((len(ALPHABET), len(ALPHABET)), dtype=torch.long)
    rows = []
    for index, char in enumerate(tqdm(ALPHABET, desc="Scoring generated classes", ascii=True)):
        images = sample_glyphs(generator, char * samples_per_class, seed + index)
        # Bound recognizer inference batches independently of the requested sample count.
        probabilities = torch.cat(
            [recognizer(batch.to(device)).softmax(1).cpu() for batch in images.split(128)]
        )
        confidence, predictions = probabilities.max(1)
        confusion[index] = torch.bincount(predictions, minlength=len(ALPHABET))
        for sample, (prediction, probability) in enumerate(
            zip(predictions, confidence, strict=True)
        ):
            rows.append(
                [char, sample + 1, ALPHABET[int(prediction)], float(probability), seed + index]
            )
    metrics = confusion_metrics(confusion)
    weak_audit_classes = [char for char in ALPHABET if audit["per_class"][char]["accuracy"] < 0.85]
    confusions = []
    for i, requested in enumerate(ALPHABET):
        for j, observed in enumerate(ALPHABET):
            if i != j and int(confusion[i, j]) > 0:
                confusions.append(
                    {"requested": requested, "predicted": observed, "count": int(confusion[i, j])}
                )
    report = {
        "kind": "recognizer-agreement-proxy",
        "generator_sha256": sha256_file(generator_checkpoint),
        "recognizer_sha256": sha256_file(recognizer_checkpoint),
        "seed": seed,
        "samples_per_class": samples_per_class,
        "generated_agreement": metrics,
        "recognizer_real_audit": audit,
        "recognizer_audit_partition": recognizer_metadata["audit_partition"],
        "classes_with_real_audit_accuracy_below_85_percent": weak_audit_classes,
        "largest_confusions": sorted(confusions, key=lambda item: -item["count"])[:30],
        "test_set_evaluated": False,
        "warning": (
            "Agreement with this imperfect classifier is not ground-truth human accuracy, "
            "sharpness or page readability. Interpret weak classes alongside real-audit errors "
            "and human review; softmax confidence is not calibrated."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "agreement.json", report)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["requested", "sample", "predicted", "uncalibrated_confidence", "class_seed"]
        )
        writer.writerows(rows)
    print(
        f"Generated exact-label agreement: {metrics['accuracy']:.2%}; "
        f"recognizer real-audit accuracy: {audit['accuracy']:.2%}. See {output / 'agreement.json'}"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/agreement-baseline-10"))
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args()
    score_generation(
        args.checkpoint,
        args.recognizer,
        args.output,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
