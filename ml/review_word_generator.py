"""Score owner-entered blinded transcriptions; never optimize or run OCR."""

import argparse
import csv
import json
from pathlib import Path

from ml.utils import sha256_file, write_json
from ml.word_reader import edit_distance


def score_review(directory, answers_path=None, *, reviewer="human"):
    directory = Path(directory)
    metadata = json.loads((directory / "review.json").read_text(encoding="utf-8"))
    unseen = metadata.get("kind") == "unseen-content"
    count, threshold = (65, 52) if unseen else (32, 31)
    if reviewer not in {"human", "agent"}:
        raise ValueError("Unknown reviewer type.")
    key_path = directory / "answer-key.csv"
    if sha256_file(key_path) != metadata["answer_key_sha256"]:
        raise ValueError("Review answer key changed.")
    with key_path.open(newline="", encoding="utf-8") as stream:
        key = list(csv.DictReader(stream))
    answers_path = (
        Path(answers_path) if answers_path is not None else directory / "transcriptions.csv"
    )
    if answers_path.suffix.lower() == ".json":
        submitted = json.loads(answers_path.read_text(encoding="utf-8"))
        if (
            submitted.get("format") != "inksy-blind-word-review-v1"
            or submitted.get("checkpoint_sha256") != metadata["checkpoint_sha256"]
        ):
            raise ValueError(
                "Answers belong to a different checkpoint or unsupported review format."
            )
        answers = submitted.get("answers", [])
        hashes = {row["sample_id"]: row["image_sha256"] for row in key}
        if not isinstance(answers, list) or any(
            not isinstance(row, dict)
            or row.get("sample_id") not in hashes
            or row.get("image_sha256") != hashes[row["sample_id"]]
            for row in answers
        ):
            raise ValueError("Submitted answers do not match the reviewed images.")
    else:
        with answers_path.open(newline="", encoding="utf-8-sig") as stream:
            answers = list(csv.DictReader(stream))
    ids = {row["sample_id"] for row in key}
    if (
        len(key) != count
        or len(ids) != count
        or len(answers) != count
        or {row["sample_id"] for row in answers} != ids
    ):
        raise ValueError(f"Review must contain all {count} sample IDs exactly once.")
    indexed = {row["sample_id"]: row for row in answers}
    results = []
    for row in key:
        sample_id = row["sample_id"]
        if not sample_id.startswith("sample-") or not sample_id[7:].isdigit():
            raise ValueError("Invalid sample ID.")
        if sha256_file(directory / (sample_id + ".png")) != row["image_sha256"]:
            raise ValueError("Review image changed.")
        answer = indexed[sample_id]
        transcript = answer.get("transcription")
        visual = answer.get("no_visual_defects")
        if not isinstance(transcript, str) or not isinstance(visual, str):
            raise ValueError("Every answer needs a text transcription and visual yes/no field.")
        transcript = transcript.strip()
        visual = visual.strip().lower()
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
        "purpose": "Unseen-content development review" if unseen else "Training-fit review",
        "reviewer": reviewer,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "exact_words": exact,
        "examples": count,
        "exact_word_accuracy": exact / count,
        "character_error_rate": sum(row["edits"] for row in results)
        / sum(len(row["requested"]) for row in results),
        "all_images_visually_acceptable": visual,
        "pilot_gate_passed" if unseen else "fit_gate_passed": (
            exact >= threshold and visual and reviewer == "human"
        ),
        "generalization_established": False,
        "rows": results,
        "transcriptions_sha256": sha256_file(answers_path),
        "transcriptions_source": str(answers_path.resolve()),
        "limitation": "Blinding relies on the reviewer not inspecting prompts/key first.",
    }
    write_json(directory / f"{reviewer}-review.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument(
        "--answers", type=Path, help="Downloaded review JSON or a transcription CSV."
    )
    parser.add_argument("--reviewer", choices=("human", "agent"), default="human")
    args = parser.parse_args()
    report = score_review(args.review, args.answers, reviewer=args.reviewer)
    gate = report.get("pilot_gate_passed", report.get("fit_gate_passed"))
    print(
        f"Exact words: {report['exact_words']}/{report['examples']}. "
        f"Human acceptance gate passed: {gate}."
    )


if __name__ == "__main__":
    main()
