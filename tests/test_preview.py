import importlib
import json
from zipfile import ZipFile

import pytest
import torch
from PIL import Image

from ml.characters import UnsupportedText
from ml.model import ConditionalVAE, ModelConfig
from ml.preview import create_preview, prepare_text, sample_occurrences
from rendering.compositor import LayoutOptions
from rendering.report import preview_html


def test_text_normalization_is_opt_in_and_unsupported_characters_are_not_dropped():
    raw = "\u201cHi\u201d\r\n\tA"
    with pytest.raises(UnsupportedText):
        prepare_text(raw)
    assert prepare_text(raw, True) == '"Hi"\n    A'
    with pytest.raises(UnsupportedText):
        prepare_text("Hi \u00e9", True)


def test_sampling_covers_each_occurrence_and_is_reproducible_without_training():
    model = ConditionalVAE(ModelConfig(latent_dim=4, label_dim=3, base_channels=4))
    state = {key: value.clone() for key, value in model.state_dict().items()}
    text = "Aa, A9!"
    first = sample_occurrences(model, text, 42, batch_size=2, show_progress=False)
    second = sample_occurrences(model, text, 42, batch_size=2, show_progress=False)
    different = sample_occurrences(model, text, 43, batch_size=2, show_progress=False)
    assert set(first) == {0, 1, 4, 5}
    assert all(first[index].tobytes() == second[index].tobytes() for index in first)
    assert any(first[index].tobytes() != different[index].tobytes() for index in first)
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_exported_pages_zip_and_manifest_describe_the_same_images(tmp_path, monkeypatch):
    """Fixture model tests export fidelity, not learned handwriting quality."""
    module = importlib.import_module("ml.preview")
    model = ConditionalVAE(ModelConfig(latent_dim=4, label_dim=3, base_channels=4)).eval()
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"synthetic model identity")
    monkeypatch.setattr(
        module,
        "load_checkpoint",
        lambda *args: (model, {"epoch": 0, "quality_status": "test-fixture"}),
    )
    text = "Hello,\n\nThis is Inksy 123!"
    output = tmp_path / "preview"
    result = create_preview(
        text,
        checkpoint=checkpoint,
        output=output,
        options=LayoutOptions(width=500, height=500, margin=32),
        show_progress=False,
    )
    assert result["training_performed"] is False
    assert result["rendered_text"] == text
    assert result["coverage"]["model_glyphs"] == 19
    assert result["coverage"]["procedural_punctuation"] == 2
    assert len(result["placements"]) == 21
    stored = json.loads((output / "manifest.json").read_text())
    assert stored["checkpoint_sha256"] == result["checkpoint_sha256"]
    with ZipFile(output / "pages.zip") as archive:
        assert archive.namelist() == [p["filename"] for p in result["pages"]]
        for page in result["pages"]:
            assert archive.read(page["filename"]) == (output / page["filename"]).read_bytes()
            with Image.open(output / page["filename"]) as image:
                assert image.size == (page["width"], page["height"])
    assert (output / "input.txt").read_text() == text
    html = (output / "preview.html").read_text(encoding="utf-8")
    assert "Original typed text" in html and "page-001.png" in html
    with pytest.raises(FileExistsError):
        create_preview(text, checkpoint=checkpoint, output=output, show_progress=False)


def test_original_input_is_escaped_in_report_and_model_characters_stay_images(
    tmp_path, monkeypatch
):
    module = importlib.import_module("ml.preview")
    model = ConditionalVAE(ModelConfig(latent_dim=4, label_dim=3, base_channels=4)).eval()
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"synthetic identity")
    monkeypatch.setattr(module, "load_checkpoint", lambda *args: (model, {}))
    result = create_preview(
        "A", checkpoint=checkpoint, output=tmp_path / "preview", show_progress=False
    )
    result["original_text"] = '<script>alert("x")</script>'
    rendered = preview_html(result)
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in rendered
    assert '<script>alert("x")</script>' not in rendered
    assert '<img src="page-001.png"' in rendered


def test_normalized_file_input_preserves_original_line_endings_in_source_export(
    tmp_path, monkeypatch
):
    module = importlib.import_module("ml.preview")
    model = ConditionalVAE(ModelConfig(latent_dim=4, label_dim=3, base_channels=4)).eval()
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"synthetic identity")
    monkeypatch.setattr(module, "load_checkpoint", lambda *args: (model, {}))
    raw = "\u201cA\u201d\r\n\tB"
    output = tmp_path / "normalized"
    result = create_preview(
        raw, checkpoint=checkpoint, output=output, apply_normalization=True, show_progress=False
    )
    assert result["normalization_applied"]
    assert result["original_text"] == raw
    assert result["rendered_text"] == '"A"\n    B'
    assert (output / "input.txt").read_bytes() == raw.encode("utf-8")
    assert b"\r\r\n" not in (output / "preview.html").read_bytes()
