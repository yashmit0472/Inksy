"""Export blind studio readability grids and score a reviewer's transcriptions."""

import argparse
import csv
import json
import random
import re
from pathlib import Path
from string import ascii_lowercase, ascii_uppercase, digits

import torch
from PIL import Image, ImageDraw, ImageOps

from ml.print_handwriting import load_letter_variants, with_print_punctuation
from ml.utils import sha256_file, write_json
from ml.word_generator_preview import write_csv
from ml.word_reader import edit_distance
from rendering.print_pages import print_pages
from rendering.studio import DEFAULT_CHECKPOINT

SEED = 20261109
COUNTS = {"words": 200, "mixed": 20, "digits": 20}


def export(records, prior_key, output):
    run = json.loads((DEFAULT_CHECKPOINT.parent / "run.json").read_text(encoding="utf-8"))
    checkpoint_hash, records_hash = sha256_file(DEFAULT_CHECKPOINT), sha256_file(records)
    if (checkpoint_hash, records_hash) != (
        run["checkpoint_sha256"],
        run["selection"]["records_sha256"],
    ):
        raise ValueError("Checkpoint or records do not match the studio training run.")
    with prior_key.open(newline="", encoding="utf-8") as stream:
        excluded = {row["requested"] for row in csv.DictReader(stream)}
    with records.open(encoding="utf-8") as stream:
        entries = [json.loads(line) for line in stream]
    excluded.update(row["text"].casefold() for row in entries if row["partition"] == "train")
    pool = {}
    for row in entries:
        word = row["text"]
        if (
            row["partition"] == "content_validation"
            and re.fullmatch("[a-z]{3,10}", word)
            and word not in excluded
        ):
            pool.setdefault(word, row["id"])
    rng = random.Random(SEED)
    words = rng.sample(sorted(pool), COUNTS["words"])
    for letter in sorted(set(ascii_lowercase) - set("".join(words))):
        word = rng.choice(sorted(word for word in pool if letter in word and word not in words))
        words.append(word)
    words = words[len(words) - COUNTS["words"] :]
    if set("".join(words)) != set(ascii_lowercase):
        raise ValueError("The benchmark must cover every lowercase letter.")
    upper = list(ascii_uppercase * 3 + "AB")
    lower = list(ascii_lowercase * 3 + "ab")
    numbers = list(digits * 12)
    for symbols in (upper, lower, numbers):
        rng.shuffle(symbols)
    mixed = []
    for start in range(0, len(upper), 4):
        symbols = upper[start : start + 4] + lower[start : start + 4]
        rng.shuffle(symbols)
        mixed.append("".join(symbols))
    prompts = {
        "words": words,
        "mixed": mixed,
        "digits": ["".join(numbers[i : i + 6]) for i in range(0, len(numbers), 6)],
    }
    torch.set_num_threads(4)
    bank, _ = load_letter_variants(DEFAULT_CHECKPOINT)
    bank = with_print_punctuation(bank)
    output.mkdir(parents=True, exist_ok=False)
    key = []
    for group, texts in prompts.items():
        assert len(texts) == COUNTS[group] and len(set(texts)) == len(texts)
        for start in range(0, len(texts), 20):
            sheet = Image.new("L", (1160, 390), 255)
            draw = ImageDraw.Draw(sheet)
            for offset, prompt in enumerate(texts[start : start + 20]):
                index = start + offset + 1
                sample = f"{group}-{index:03d}"
                (page,) = print_pages(prompt, bank, seed=SEED + len(key))
                image = page.crop(ImageOps.invert(page).getbbox())
                assert image.width <= 280 and image.height <= 52
                x, y = offset % 4 * 290 + 8, offset // 4 * 78 + 6
                sheet.paste(image, (x, y))
                draw.text((x, y + 54), sample, fill=0)
                key.append(
                    {
                        "sample": sample,
                        "group": group,
                        "prompt": prompt,
                        "source_id": pool[prompt] if group == "words" else "synthetic",
                    }
                )
            sheet.save(output / f"blind-{group}-{start // 20 + 1:02d}.png")
    key_path = output / "answer-key.csv"
    write_csv(key_path, key, ["sample", "group", "prompt", "source_id"])
    write_csv(
        output / "blank-transcriptions.csv",
        [{"sample": row["sample"], "transcription": ""} for row in key],
        ["sample", "transcription"],
    )
    write_json(
        output / "benchmark.json",
        {
            "checkpoint_sha256": checkpoint_hash,
            "records_sha256": records_hash,
            "answer_key_sha256": sha256_file(key_path),
            "prior_key_sha256": sha256_file(prior_key),
            "seed": SEED,
            "counts": COUNTS,
            "word_partition": "content_validation",
            "synthetic_groups": ["mixed", "digits"],
        },
    )


def score(output, answers, reviewer):
    metadata = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    key_path = output / "answer-key.csv"
    if metadata["answer_key_sha256"] != sha256_file(key_path):
        raise ValueError("Answer key changed after export.")
    with key_path.open(newline="", encoding="utf-8") as stream:
        key = list(csv.DictReader(stream))
    with answers.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    submitted = {row["sample"]: row["transcription"].strip() for row in rows}
    if len(rows) != len(submitted) or set(submitted) != {row["sample"] for row in key}:
        raise ValueError("Supply one transcription for every sample.")
    if any(not answer for answer in submitted.values()):
        raise ValueError("Use <unreadable> for any sample you cannot read.")
    for row in key:
        answer = submitted[row["sample"]]
        row["transcription"] = "" if answer == "<unreadable>" else answer
        row["edits"] = edit_distance(row["prompt"], row["transcription"])
    metrics = {}
    for group in COUNTS:
        group_rows = [row for row in key if row["group"] == group]
        exact = sum(row["edits"] == 0 for row in group_rows)
        edits = sum(row["edits"] for row in group_rows)
        chars = sum(len(row["prompt"]) for row in group_rows)
        metrics[group] = {
            "exact": exact,
            "count": len(group_rows),
            "edits": edits,
            "characters": chars,
            "character_error_rate": edits / chars,
        }
        print(f"{group}: {exact}/{len(group_rows)} exact, {edits}/{chars} character edits")
    write_json(
        output / f"{reviewer}-review.json",
        {
            **metadata,
            "reviewer": reviewer,
            "transcriptions_sha256": sha256_file(answers),
            "metrics": metrics,
            "samples": key,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--records", type=Path, default=Path("artifacts/iam-words/records.jsonl"))
    parser.add_argument(
        "--prior-key", type=Path, default=Path("artifacts/studio-readability-review/answer-key.csv")
    )
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--reviewer", choices=("human", "agent"), default="human")
    args = parser.parse_args()
    if args.answers is None:
        export(args.records, args.prior_key, args.output)
    else:
        score(args.output, args.answers, args.reviewer)


if __name__ == "__main__":
    main()
