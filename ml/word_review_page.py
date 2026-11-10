"""Build a portable, offline blind-review form from existing generated word PNGs."""

import argparse
import base64
import csv
import html
import json
import re
from pathlib import Path

from ml.utils import sha256_file


def build_page(review, output):
    review, output = Path(review), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new review page path.")
    metadata = json.loads((review / "review.json").read_text(encoding="utf-8"))
    if not re.fullmatch(r"[0-9a-f]{64}", metadata.get("checkpoint_sha256", "")):
        raise ValueError("Invalid checkpoint identity.")
    key_path = review / "answer-key.csv"
    if sha256_file(key_path) != metadata["answer_key_sha256"]:
        raise ValueError("Review answer key changed.")
    with key_path.open(newline="", encoding="utf-8") as stream:
        keys = list(csv.DictReader(stream))
    if len(keys) != 32 or len({row["sample_id"] for row in keys}) != 32:
        raise ValueError("Expected 32 unique review samples.")
    cards, samples = [], []
    for row in keys:
        identifier = row["sample_id"]
        if not re.fullmatch(r"sample-\d{3}", identifier):
            raise ValueError("Invalid review sample ID.")
        path = review / (identifier + ".png")
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError("Review image changed.")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        # Do not serialize the requested word, source ID, answer key or original filenames.
        samples.append({"sample_id": identifier, "image_sha256": row["image_sha256"]})
        cards.append(f'''<section class="card" data-sample="{identifier}">
<h2>Sample {html.escape(identifier[7:])}</h2>
<div class="image"><img src="data:image/png;base64,{encoded}"
alt="Handwriting sample {html.escape(identifier[7:])}" height="64"></div>
<label for="{identifier}-text">What word do you read?</label>
<input id="{identifier}-text" class="transcription" type="text" maxlength="100"
autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false">
<label class="check"><input type="checkbox" class="unreadable"> I cannot read this word</label>
<label for="{identifier}-visual">Are the strokes visually acceptable?</label>
<select id="{identifier}-visual" class="visual"><option value="">Choose an answer</option>
<option value="yes">Yes — no visible defects</option>
<option value="no">No — missing, smeared or clipped strokes</option></select>
</section>''')
    payload = json.dumps({"checkpoint_sha256": metadata["checkpoint_sha256"], "samples": samples})
    template = Path(__file__).with_name("word_review_page.html").read_text(encoding="utf-8")
    page = template.replace("__CARDS__", "\n".join(cards)).replace("__REVIEW_DATA__", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_page(args.review, args.output))


if __name__ == "__main__":
    main()
