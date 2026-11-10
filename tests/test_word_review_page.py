import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from ml.review_word_generator import score_review
from ml.utils import sha256_file, write_json
from ml.word_generator_preview import write_csv
from ml.word_review_page import build_page


@pytest.fixture
def review_fixture(tmp_path):
    review = tmp_path / "review"
    review.mkdir()
    rows = []
    for index in range(1, 33):
        identifier = f"sample-{index:03d}"
        path = review / f"{identifier}.png"
        Image.new("L", (64, 64), 255).save(path)
        rows.append(
            {
                "sample_id": identifier,
                "requested": f"HIDDEN_ANSWER_{index}",
                "source_id": f"HIDDEN_SOURCE_{index}",
                "image_sha256": sha256_file(path),
            }
        )
    write_csv(review / "answer-key.csv", rows, list(rows[0]))
    write_json(
        review / "review.json",
        {
            "checkpoint_sha256": "b" * 64,
            "answer_key_sha256": sha256_file(review / "answer-key.csv"),
        },
    )
    return review, rows


def test_portable_review_does_not_embed_ground_truth_or_external_resources(
    review_fixture, tmp_path
):
    review, _ = review_fixture
    page = build_page(review, tmp_path / "review.html")
    source = page.read_text(encoding="utf-8")
    assert source.count('src="data:image/png;base64,') == 32
    assert "HIDDEN_ANSWER" not in source and "HIDDEN_SOURCE" not in source
    assert 'src="http' not in source and 'href="http' not in source
    data = json.loads(re.search(r'<script id="review-data"[^>]*>(.*?)</script>', source, re.S)[1])
    assert len(data["samples"]) == 32
    assert set(data["samples"][0]) == {"sample_id", "image_sha256"}
    with pytest.raises(FileExistsError):
        build_page(review, page)


def test_json_answers_are_bound_to_checkpoint_and_images(review_fixture, tmp_path):
    review, rows = review_fixture
    # Perfect synthetic answers exercise scoring; this is not a model evaluation.
    payload = {
        "format": "inksy-blind-word-review-v1",
        "checkpoint_sha256": "b" * 64,
        "answers": [
            {
                "sample_id": row["sample_id"],
                "image_sha256": row["image_sha256"],
                "transcription": row["requested"],
                "no_visual_defects": "yes",
            }
            for row in rows
        ],
    }
    path = tmp_path / "answers.json"
    write_json(path, payload)
    report = score_review(review, path)
    assert report["exact_words"] == 32 and report["fit_gate_passed"]
    assert not report["generalization_established"]
    payload["checkpoint_sha256"] = "c" * 64
    write_json(path, payload)
    with pytest.raises(ValueError, match="different checkpoint"):
        score_review(review, path)
    payload["checkpoint_sha256"] = "b" * 64
    payload["answers"][0]["image_sha256"] = "c" * 64
    write_json(path, payload)
    with pytest.raises(ValueError, match="reviewed images"):
        score_review(review, path)


def test_page_rejects_corrupt_images_or_injected_metadata(review_fixture, tmp_path):
    review, _ = review_fixture
    metadata = json.loads((review / "review.json").read_text())
    metadata["checkpoint_sha256"] = "</script><script>alert(1)</script>"
    write_json(review / "review.json", metadata)
    with pytest.raises(ValueError, match="checkpoint identity"):
        build_page(review, tmp_path / "bad.html")
    metadata["checkpoint_sha256"] = "b" * 64
    write_json(review / "review.json", metadata)
    (review / "sample-001.png").write_bytes(b"bad")
    with pytest.raises(ValueError, match="image changed"):
        build_page(review, tmp_path / "bad.html")


@pytest.mark.skipif(shutil.which("node") is None, reason="Optional JS form harness needs Node")
def test_form_progress_draft_restore_and_export_without_answers_in_page(review_fixture, tmp_path):
    review, _ = review_fixture
    page = build_page(review, tmp_path / "form.html")
    result = subprocess.run(
        [
            shutil.which("node"),
            str(Path(__file__).with_name("word_review_form_harness.cjs")),
            str(page),
            str(tmp_path / "export.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "form checks passed" in result.stdout
    # The harness deliberately marks every image unreadable. Never infer a real quality score.
    report = score_review(review, tmp_path / "export.json")
    assert report["exact_words"] == 0 and not report["fit_gate_passed"]
