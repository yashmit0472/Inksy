"""Export the frozen unseen-word prompts for review using predicted widths."""

import argparse
import json
import random
from pathlib import Path

import torch

from ml.print_handwriting import compose_word, load_letters
from ml.utils import sha256_file, write_json
from ml.word_generator import encode_texts, load_generator
from ml.word_generator_preview import generate_images, save_grid, write_csv


def export(run, output, writer, baseline, print_checkpoint=None):
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    checkpoint = print_checkpoint or run / "best.pt"
    digest = sha256_file(checkpoint)
    words = json.loads((run / "holdout-prompts.json").read_text(encoding="utf-8"))
    if len(words) != 65 or len(set(words)) != len(words):
        raise ValueError("Expected 65 distinct frozen prompts.")
    records = Path(provenance["settings"]["prepared"]) / "records.jsonl"
    if sha256_file(records) != provenance["records_sha256"]:
        raise ValueError("Prepared data digest changed.")
    ids = set(provenance["partitions"]["content_validation"]["ids"])
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    frozen_words = {
        row["text"] for row in rows if row["id"] in ids and row["partition"] == "content_validation"
    }
    if set(words) != frozen_words or any(
        row["partition"] == "train" and row["text"].casefold() in frozen_words for row in rows
    ):
        raise ValueError("Evaluation prompts differ from the training-unseen partition.")
    encode_texts(words)
    random.Random(provenance["seed"]).shuffle(words)
    if print_checkpoint:
        fit = json.loads((checkpoint.parent / "run.json").read_text(encoding="utf-8"))
        if digest != fit["checkpoint_sha256"] or (
            fit["selection"]["records_sha256"] != provenance["records_sha256"]
        ):
            raise ValueError("Print checkpoint or training dataset identity differs.")
        letters = load_letters(checkpoint)
        images = [compose_word(word, letters) for word in words]
        style_label = "Learned print glyphs"
    else:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        if digest != summary["best_checkpoint_sha256"]:
            raise ValueError("Selected checkpoint digest differs from the run summary.")
        model, _ = load_generator(checkpoint)
        images = generate_images(model, words, writer=writer)
        style_label = f"Candidate | writer {writer}"
    old_model, _ = load_generator(baseline)
    old_images = generate_images(old_model, words)
    output.mkdir(parents=True, exist_ok=False)
    labels = [f"sample-{i:03d}" for i in range(1, len(words) + 1)]
    rows = []
    for label, word, image in zip(labels, words, images, strict=True):
        path = output / f"{label}.png"
        image.save(path)
        rows.append({"sample_id": label, "requested": word, "image_sha256": sha256_file(path)})
    for start in range(0, len(words), 16):
        end = start + 16
        save_grid(
            output / f"blind-{start // 16 + 1}.png",
            images[start:end],
            labels[start:end],
            "Generated unseen words | read before opening the answer key",
        )
    save_grid(output / "candidate-labeled.png", images, words, style_label)
    save_grid(output / "prior-labeled.png", old_images, words, "Prior generator | writer 000")
    key = output / "answer-key.csv"
    write_csv(key, rows, ["sample_id", "requested", "image_sha256"])
    fields = ["sample_id", "transcription", "no_visual_defects"]
    write_csv(
        output / "transcriptions.csv",
        [dict(zip(fields, [label, "", ""], strict=True)) for label in labels],
        fields,
    )
    write_json(
        output / "review.json",
        {
            "kind": "unseen-content",
            "writer_id": None if print_checkpoint else writer,
            "rendering": "learned print glyph composition" if print_checkpoint else "whole word",
            "examples": 65,
            "exact_required": 52,
            "checkpoint_sha256": digest,
            "baseline_sha256": sha256_file(baseline),
            "answer_key_sha256": sha256_file(key),
            "target_width_used": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/multi-writer-pilot"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/multi-writer-pilot-review"))
    parser.add_argument("--writer", default="150")
    parser.add_argument("--print-checkpoint", type=Path)
    parser.add_argument(
        "--baseline", type=Path, default=Path("artifacts/word-generator-detail-coordinate/best.pt")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    export(args.run, args.output, args.writer, args.baseline, args.print_checkpoint)


if __name__ == "__main__":
    main()
