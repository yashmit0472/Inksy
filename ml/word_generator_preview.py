"""Inference-only word exports and blinded training-fit review materials."""

import argparse
import csv
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from ml.utils import sha256_file, write_json
from ml.word_generator import load_generator


def paper_image(pixels):
    if not torch.isfinite(pixels).all():
        raise FloatingPointError("Cannot export non-finite pixels.")
    return Image.fromarray(
        ((1 - pixels.detach().cpu().squeeze(0)).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    )


@torch.inference_mode()
def generate_images(model, texts, widths=None, batch_size=8):
    model.eval()
    images = []
    for start in range(0, len(texts), batch_size):
        batch_widths = None if widths is None else widths[start : start + batch_size]
        result = model(texts[start : start + batch_size], batch_widths)
        for pixels, width in zip(result["logits"].sigmoid(), result["widths"], strict=True):
            images.append(paper_image(pixels[:, :, : int(width)]))
    return images


def save_grid(path, images, labels, title):
    if not images or len(images) != len(labels):
        raise ValueError("A grid requires one label per image.")
    columns, cell_width, cell_height = 4, 400, 90
    sheet = Image.new(
        "RGB", (columns * cell_width, 44 + ((len(images) + 3) // 4) * cell_height), "white"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill="black")
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        x, y = (index % columns) * cell_width + 8, (index // columns) * cell_height + 44
        sheet.paste(image, (x, y))
        draw.text((x, y + 66), label, fill="black")
    sheet.save(path)


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_fit_review(model, items, output, *, seed, checkpoint_sha256):
    """No scores are inferred from reconstruction loss or the weak OCR model."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    texts = [item["text"] for item in items]
    widths = torch.tensor([item["image"].shape[-1] for item in items])
    predicted = generate_images(model, texts)
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    labels = [f"sample-{index + 1:03d}" for index in range(len(order))]
    blind = [predicted[index] for index in order]
    save_grid(
        output / "blind-predicted-width.png",
        blind,
        labels,
        "GENERATED TRAINING-FIT DIAGNOSTIC | predicted widths | transcribe before opening key",
    )
    key, answers = [], []
    for label, index, image in zip(labels, order, blind, strict=True):
        filename = label + ".png"
        image.save(output / filename)
        key.append(
            {
                "sample_id": label,
                "requested": texts[index],
                "source_id": items[index]["id"],
                "image_sha256": sha256_file(output / filename),
            }
        )
        answers.append({"sample_id": label, "transcription": "", "no_visual_defects": ""})
    write_csv(
        output / "answer-key.csv", key, ["sample_id", "requested", "source_id", "image_sha256"]
    )
    write_csv(
        output / "transcriptions.csv", answers, ["sample_id", "transcription", "no_visual_defects"]
    )
    save_grid(
        output / "real-targets.png",
        [paper_image(item["image"]) for item in items],
        texts,
        "REAL IAM TRAINING REFERENCES | not generated",
    )
    save_grid(
        output / "generated-target-width.png",
        generate_images(model, texts, widths),
        texts,
        "GENERATED FIT DIAGNOSTIC | ground-truth widths supplied | not end-to-end",
    )
    save_grid(
        output / "generated-predicted-width.png",
        predicted,
        texts,
        "GENERATED TRAINING WORDS | predicted widths | memorization is not generalization",
    )
    controls = [
        texts[0],
        texts[0][::-1],
        "".join(sorted(texts[0])),
        "letter",
        "little",
        "letters",
        "tests",
        "test",
    ]
    save_grid(
        output / "conditioning-controls.png",
        generate_images(model, controls),
        controls,
        "GENERATED CONDITIONING CONTROLS | no accuracy claim; may overlap training vocabulary",
    )
    write_json(
        output / "review.json",
        {
            "purpose": "32-word training-fit diagnostic; not unseen-word evaluation",
            "checkpoint_sha256": checkpoint_sha256,
            "seed": seed,
            "examples": len(items),
            "inference_uses_target_width": False,
            "quality_status": "awaiting-human-review",
            "answer_key_sha256": sha256_file(output / "answer-key.csv"),
            "exact_required": 31,
            "visual_review_required": True,
            "generalization_established": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--text", nargs="+", required=True, help="Lowercase words, 2–10 letters each."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError("Choose a new preview directory.")
    model, checkpoint = load_generator(args.checkpoint)
    images = generate_images(model, args.text)
    args.output.mkdir(parents=True)
    for index, image in enumerate(images):
        image.save(args.output / f"word-{index + 1:03d}.png")
    save_grid(
        args.output / "words.png",
        images,
        args.text,
        "GENERATED WORDS | predicted widths | experimental single-writer model",
    )
    write_json(
        args.output / "provenance.json",
        {
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "texts": args.text,
            "model_provenance": checkpoint["provenance"],
            "target_images_or_widths_used": False,
            "quality_status": "unreviewed",
        },
    )
    print(args.output / "words.png")


if __name__ == "__main__":
    main()
