"""Score owner-entered blinded transcriptions; never optimize or run OCR."""

import argparse
import csv
import json
from pathlib import Path

from ml.utils import sha256_file, write_json
from ml.word_reader import edit_distance


def score_review(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "review.json").read_text(encoding="utf-8"))
    key_path = directory / "answer-key.csv"
    if sha256_file(key_path) != metadata["answer_key_sha256"]:
        raise ValueError("Review answer key changed.")
    with key_path.open(newline="", encoding="utf-8") as stream:
        key = list(csv.DictReader(stream))
    with (directory / "transcriptions.csv").open(newline="", encoding="utf-8") as stream:
        answers = list(csv.DictReader(stream))
    ids = {row["sample_id"] for row in key}
    if (
        len(key) != 32
        or len(ids) != 32
        or len(answers) != 32
        or {row["sample_id"] for row in answers} != ids
    ):
        raise ValueError("Review must contain all 32 sample IDs exactly once.")
    indexed = {row["sample_id"]: row for row in answers}
    results = []
    for row in key:
        sample_id = row["sample_id"]
        if not sample_id.startswith("sample-") or not sample_id[7:].isdigit():
            raise ValueError("Invalid sample ID.")
        if sha256_file(directory / (sample_id + ".png")) != row["image_sha256"]:
            raise ValueError("Review image changed.")
        answer = indexed[sample_id]
        transcript = answer["transcription"].strip()
        visual = answer["no_visual_defects"].strip().lower()
        if not transcript or visual not in {"yes", "no"}:
            raise ValueError("Fill every transcription (use <unreadable>) and visual yes/no field.")
        if transcript == "<unreadable>":
            transcript = ""
        results.append(
            {
                "sample_id": sample_id,
                "requested": row["requested"],
                "transcription": transcript,
                "exact": transcript == row["requested"],
                "edits": edit_distance(row["requested"], transcript),
                "no_visual_defects": visual == "yes",
            }
        )
    exact = sum(row["exact"] for row in results)
    visual = all(row["no_visual_defects"] for row in results)
    report = {
        "purpose": "Human-reviewed training-fit only",
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "exact_words": exact,
        "examples": 32,
        "exact_word_accuracy": exact / 32,
        "character_error_rate": sum(row["edits"] for row in results)
        / sum(len(row["requested"]) for row in results),
        "all_images_visually_acceptable": visual,
        "fit_gate_passed": exact >= 31 and visual,
        "generalization_established": False,
        "rows": results,
        "transcriptions_sha256": sha256_file(directory / "transcriptions.csv"),
        "limitation": "Blinding relies on the reviewer not inspecting prompts/key first.",
    }
    write_json(directory / "human-review.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    report = score_review(args.review)
    print(
        f"Training-fit exact: {report['exact_words']}/32. "
        f"Fit gate passed: {report['fit_gate_passed']}. Generalization remains untested."
    )


if __name__ == "__main__":
    main()
