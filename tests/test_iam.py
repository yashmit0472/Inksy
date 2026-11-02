import io
import json
import tarfile
import zipfile

import pytest
import torch
from PIL import Image, ImageDraw

from ml.iam import (
    WORD_TOKENS,
    IAMConfig,
    IAMWordDataset,
    assign_partition,
    collate_words,
    ctc_min_steps,
    encode_word,
    ensure_training_styles,
    normalized_width,
    prepare,
    preprocess_word,
    read_forms,
    read_splits,
    read_words,
)
from ml.iam_download import md5_file
from ml.iam_import import extract_splits, extract_tar, safe_relative


@pytest.fixture
def iam_fixture(tmp_path):
    """Invented metadata and stroke masks; never represent these as real IAM samples."""
    root = tmp_path / "iam"
    (root / "ascii").mkdir(parents=True)
    (root / "splits").mkdir()
    forms = ["a01-000u", "a01-001u", "a01-002u", "a01-003u"]
    (root / "ascii/forms.txt").write_text(
        "# fixture\n" + "\n".join(f"{form} {i:03d} 1 all 1 1 1 1" for i, form in enumerate(forms))
    )
    entries = []
    for index in range(60):
        identifier = f"{forms[0]}-{index:02d}-00"
        entries.append(f"{identifier} ok 154 1 2 32 16 NN word{index}")
        path = root / "words/a01" / forms[0] / f"{identifier}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("L", (32 + index % 7, 16), 255)
        ImageDraw.Draw(image).line((3, 12, 8, 3, 15, 12, 24, 5), fill=40, width=2)
        image.save(path)
    for form in forms[1:]:
        entries.append(f"{form}-00-00 ok 154 1 2 32 16 NN reserved")
    (root / "ascii/words.txt").write_text("# fixture\n" + "\n".join(entries))
    names = ["trainset.txt", "validationset1.txt", "validationset2.txt", "testset.txt"]
    for index, name in enumerate(names):
        lines = [f"{forms[0]}-{i:02d}" for i in range(60)] if index == 0 else [f"{forms[index]}-00"]
        (root / "splits" / name).write_text("\n".join(lines))
    return root


def test_real_word_format_preserves_case_status_and_writer_identity(tmp_path):
    forms_path, words_path = tmp_path / "forms.txt", tmp_path / "words.txt"
    forms_path.write_text("# header\na01-000u 007 2 prt 7 5 52 36\n")
    words_path.write_text(
        "a01-000u-00-00 ok 154 408 768 27 51 AT Mr.\na01-000u-00-01 err 154 -1 -1 0 0 NN Bad\n"
    )
    rows = read_words(words_path, read_forms(forms_path))
    assert rows[0]["writer_id"] == "007" and rows[0]["text"] == "Mr."
    assert rows[1]["segmentation"] == "err"


def test_preparation_preserves_official_holdouts_and_never_opens_reserved_pixels(
    iam_fixture, tmp_path
):
    output = tmp_path / "prepared"
    manifest = prepare(iam_fixture, output)
    assert manifest["official_writer_counts"] == {
        "official_train": 1,
        "writer_validation1": 1,
        "writer_validation2": 1,
        "test": 1,
    }
    assert not manifest["test_pixels_opened"]
    assert not manifest["unknown_writer_validation_pixels_opened"]
    assert manifest["content_validation_casefold_overlap_with_train"] == 0
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    train_vocab = {row["text"].casefold() for row in rows if row["partition"] == "train"}
    content_vocab = {
        row["text"].casefold() for row in rows if row["partition"] == "content_validation"
    }
    assert train_vocab and content_vocab and not train_vocab & content_vocab
    for row in rows:
        if row["partition"] in {"test", "writer_validation1", "writer_validation2"}:
            assert row["image_sha256"] is None and row["image"] is None
    assert (output / "real-training-words.png").is_file()
    with pytest.raises(FileExistsError):
        prepare(iam_fixture, output)


def test_variable_width_batches_preserve_pixels_and_pad_with_paper(iam_fixture, tmp_path):
    output = tmp_path / "prepared"
    prepare(iam_fixture, output)
    data = IAMWordDataset(output)
    examples = [data[0], data[1], data[2]]
    batch = collate_words(examples)
    assert batch["images"].shape[:3] == (3, 1, 64)
    assert batch["images"].max() > 0 and batch["images"].min() == 0
    for index, item in enumerate(examples):
        width = item["image"].shape[-1]
        assert torch.equal(batch["images"][index, :, :, :width], item["image"])
        assert not batch["images"][index, :, :, width:].any()
    assert (batch["widths"] % 8 == 0).all()
    assert not (batch["ctc_targets"] == 0).any()
    assert batch["tokens"].shape[0] == 3
    for partition in ("test", "writer_validation1", "writer_validation2"):
        with pytest.raises(ValueError, match="Reserved"):
            IAMWordDataset(output, partition)


def test_source_image_and_prepared_record_tampering_are_detected(iam_fixture, tmp_path):
    output = tmp_path / "prepared"
    prepare(iam_fixture, output)
    data = IAMWordDataset(output)
    path = iam_fixture / data.rows[0]["image"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source image changed"):
        data[0]
    path = output / "records.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="records changed"):
        IAMWordDataset(output)


def test_casefolded_word_holdout_is_global_and_line_holdout_is_grouped():
    config = IAMConfig()
    lines = {"a01-000u-00": "official_train", "a01-000u-01": "official_train"}
    first = {"line_id": "a01-000u-00", "text": "Hello"}
    other = {"line_id": "a01-000u-01", "text": "HELLO"}
    a, b = assign_partition(first, lines, config), assign_partition(other, lines, config)
    assert (a == "content_validation") == (b == "content_validation")
    rows = [{"line_id": "a01-000u-00", "text": f"word{i}"} for i in range(100)]
    remaining = {assign_partition(row, lines, config) for row in rows} - {"content_validation"}
    assert len(remaining) == 1


def test_rare_writer_keeps_a_training_line_without_leaking_unseen_word_content():
    rows = [
        {"writer_id": "006", "line_id": "a01-011x-08", "partition": "seen_validation"},
        {"writer_id": "006", "line_id": "a01-011x-08", "partition": "seen_validation"},
        {"writer_id": "006", "line_id": "a01-011x-08", "partition": "content_validation"},
    ]
    promotions = ensure_training_styles(rows, 42)
    assert [row["partition"] for row in rows] == ["train", "train", "content_validation"]
    assert promotions == [{"writer_id": "006", "line_id": "a01-011x-08", "words": 2}]
    with pytest.raises(ValueError, match="held-out vocabulary"):
        ensure_training_styles([rows[2].copy()], 42)


def test_corrupt_status_bad_boxes_and_unsupported_text_are_logged(iam_fixture, tmp_path):
    path = iam_fixture / "ascii/words.txt"
    text = path.read_text().replace("word0\n", "word0/\n")
    text = text.replace("a01-000u-01-00 ok", "a01-000u-01-00 err")
    text += "\na01-000u-02-01 ok 154 0 0 0 0 NN invalid"
    path.write_text(text)
    manifest = prepare(iam_fixture, tmp_path / "prepared")
    assert manifest["exclusion_counts"]["unsupported_characters"] == 1
    assert manifest["exclusion_counts"]["segmentation_error"] == 1
    assert manifest["exclusion_counts"]["invalid_bounding_box"] == 1


def test_overlap_in_official_writers_or_line_ids_is_rejected(iam_fixture):
    forms = read_forms(iam_fixture / "ascii/forms.txt")
    path = iam_fixture / "splits/testset.txt"
    path.write_text("a01-000u-00")
    with pytest.raises(ValueError, match="overlapping"):
        read_splits(iam_fixture / "splits", forms)


def test_missing_source_image_fails_before_publishing_a_manifest(iam_fixture, tmp_path):
    next((iam_fixture / "words").rglob("*.png")).unlink()
    output = tmp_path / "prepared"
    with pytest.raises(FileNotFoundError):
        prepare(iam_fixture, output)
    assert not (output / "manifest.json").exists()


def test_preprocessing_preserves_aspect_ratio_and_refuses_truncation():
    config = IAMConfig()
    image = Image.new("L", (80, 40), 255)
    ImageDraw.Draw(image).rectangle((10, 10, 30, 30), fill=0)
    pixels = preprocess_word(image, config)
    assert pixels.shape == (1, 64, normalized_width(image.size, config))
    assert not pixels[:, :4].any() and not pixels[:, -4:].any()
    with pytest.raises(ValueError, match="cropping"):
        preprocess_word(Image.new("L", (1000, 10)), config)
    assert ctc_min_steps("letter") == 7
    assert encode_word("Aa9.").tolist() == [WORD_TOKENS[c] for c in "Aa9."]
    with pytest.raises(ValueError):
        encode_word("two words")


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "C:/drive",
        "x\\escape",
        "x:ads",
        "NUL",
        "a/COM1.txt",
        "trailing.",
        "trailing ",
    ],
)
def test_archive_paths_cannot_escape_or_use_windows_devices(name):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_relative(name)


def test_archive_import_copies_bytes_but_rejects_links_and_oversize(tmp_path):
    archive = tmp_path / "good.tar"
    with tarfile.open(archive, "w") as target:
        info = tarfile.TarInfo("a01/word.png")
        info.size = 5
        target.addfile(info, io.BytesIO(b"bytes"))
    result = extract_tar(archive, tmp_path / "good")
    assert result == {"files": 1, "unpacked_bytes": 5}
    assert (tmp_path / "good/a01/word.png").read_bytes() == b"bytes"
    with pytest.raises(ValueError, match="size limits"):
        extract_tar(archive, tmp_path / "too-big", max_bytes=4)
    with tarfile.open(tmp_path / "bad.tar", "w") as target:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../outside"
        target.addfile(info)
    with pytest.raises(ValueError, match="links"):
        extract_tar(tmp_path / "bad.tar", tmp_path / "bad")


def test_zip_paths_and_local_checksum_verification(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as target:
        target.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError):
        extract_splits(path, tmp_path / "splits")
    path = tmp_path / "fixture.txt"
    path.write_bytes(b"abc")
    assert md5_file(path) == "900150983cd24fb0d6963f7d28e17f72"
