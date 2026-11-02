import csv
import json
from dataclasses import asdict

import pytest
import torch
from PIL import Image

from ml.iam import WORD_ALPHABET, WORD_CONTRACT, WORD_PREPROCESSING, IAMConfig
from ml.review_word_generator import score_review
from ml.train_word_generator import FitConfig, preflight, prepare
from ml.utils import sha256_file, write_json
from ml.word_generator import (
    WordGenerator,
    WordGeneratorConfig,
    adapt_generator,
    encode_texts,
    load_generator,
    paired_loss,
    save_generator,
)
from ml.word_generator_data import load_scope
from ml.word_generator_preview import export_fit_review, generate_images, write_csv


def small_model():
    torch.manual_seed(73)
    return WordGenerator(WordGeneratorConfig(embedding=8, hidden=8, channels=16))


@pytest.mark.parametrize(
    "texts", [[], ["A"], ["Hello"], ["a"], ["two words"], ["word!"], ["éé"], ["abcdefghijk"]]
)
def test_scope_rejects_unsupported_text_without_substitution(texts):
    with pytest.raises(ValueError):
        encode_texts(texts)


def test_word_order_and_batch_padding_do_not_change_individual_output():
    model = small_model().eval()
    with torch.inference_mode():
        alone = model(["ab"], torch.tensor([48]))
        batch = model(["ab", "handwriting"[:10]], torch.tensor([48, 160]))
        reversed_text = model(["ba"], torch.tensor([48]))
    assert alone["logits"].shape == (1, 1, 64, 48)
    torch.testing.assert_close(
        alone["logits"], batch["logits"][:1, :, :, :48], atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(alone["predicted_widths"], batch["predicted_widths"][:1])
    assert not torch.allclose(alone["logits"], reversed_text["logits"])
    assert (batch["logits"][0, :, :, 48:] == -30).all()
    # Text-only inference selects its own valid widths, without a data loader.
    generated = generate_images(model, ["ab", "letters"])
    assert all(image.height == 64 and 8 <= image.width <= 384 for image in generated)


@pytest.mark.parametrize(
    "widths",
    [
        torch.tensor([0]),
        torch.tensor([49]),
        torch.tensor([392]),
        torch.tensor([48.0]),
        torch.tensor([48, 48]),
    ],
)
def test_invalid_canvas_contract(widths):
    with pytest.raises(ValueError, match="Widths"):
        small_model()(["ab"], widths)


def test_finite_gradients_reach_text_width_and_image_layers_without_weight_updates():
    model = small_model()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    widths = torch.tensor([48, 64])
    targets = torch.zeros(2, 1, 64, 64)
    targets[0, :, 12:50, 12:18] = 1
    targets[1, :, 22:26, 12:52] = 1
    result = model(["ab", "letter"], widths)
    losses = paired_loss(result, targets, widths)
    losses["total"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for parameter in (model.embedding.weight, model.width_head[-1].weight, model.output.weight):
        assert parameter.grad.abs().sum() > 0
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_loss_ignores_padding_and_penalizes_blank_ink():
    widths = torch.tensor([48])
    targets = torch.zeros(1, 1, 64, 48)
    targets[:, :, 20:44, 12:36] = 1
    correct = {"logits": targets * 20 - 10, "widths": widths, "predicted_widths": widths.float()}
    blank = {**correct, "logits": torch.full_like(targets, -10)}
    right = paired_loss(correct, targets, widths)
    wrong = paired_loss(blank, targets, widths)
    assert right["total"] < wrong["total"]
    assert wrong["foreground_mae"] > 0.99
    padded_targets = torch.nn.functional.pad(targets, (0, 16), value=0.7)
    padded = {**correct, "logits": torch.nn.functional.pad(correct["logits"], (0, 16), value=9)}
    padded_loss = paired_loss(padded, padded_targets, widths)
    for name in right:
        torch.testing.assert_close(right[name], padded_loss[name])


def test_checkpoint_roundtrip_role_and_nonfinite_rejection(tmp_path):
    model = small_model()
    path = tmp_path / "generator.pt"
    save_generator(path, model, {"writer_id": "000", "records_sha256": "a" * 64})
    loaded, metadata = load_generator(path)
    with torch.inference_mode():
        torch.testing.assert_close(model(["letters"])["logits"], loaded(["letters"])["logits"])
    metadata["role"] = "training-content-reader"
    torch.save(metadata, path)
    with pytest.raises(ValueError, match="role"):
        load_generator(path)
    save_generator(path, model, {"writer_id": "000", "records_sha256": "a" * 64})
    metadata = torch.load(path, weights_only=True)
    metadata["state_dict"]["output.bias"][0] = float("nan")
    torch.save(metadata, path)
    with pytest.raises(ValueError, match="non-finite"):
        load_generator(path)


def make_prepared(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    image_path = root / "synthetic.png"
    Image.new("L", (40, 24), 200).save(image_path)
    rows = []
    for index in range(40):
        rows.append(
            {
                "id": f"fake-{index}",
                "writer_id": "000",
                "writer_index": 0,
                "text": chr(97 + index // 26) + chr(97 + index % 26),
                "partition": "train",
                "image": image_path.name,
                "image_sha256": sha256_file(image_path),
                "width": 104,
            }
        )
    rows.extend(
        [
            {**rows[0], "id": "held-out", "text": "heldout", "partition": "content_validation"},
            {**rows[0], "id": "reserved", "text": "reserved", "partition": "test"},
            {**rows[0], "id": "other-writer", "text": "other", "writer_id": "001"},
        ]
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "records.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    write_json(
        prepared / "manifest.json",
        {
            "root": str(root),
            "contract": WORD_CONTRACT,
            "preprocessing": WORD_PREPROCESSING,
            "alphabet": WORD_ALPHABET,
            "config": asdict(IAMConfig()),
            "records_sha256": sha256_file(prepared / "records.jsonl"),
        },
    )
    return prepared


def test_scope_is_training_only_deterministic_and_hash_pinned(tmp_path):
    prepared = make_prepared(tmp_path)
    data, selected, eligible, identity = load_scope(prepared)
    _, again, _, _ = load_scope(prepared)
    assert selected == again and len(selected) == 32 and len(eligible) == 40
    assert len(set(identity["diagnostic_texts"])) == 32
    assert all(
        data.rows[index]["partition"] == "train" and data.rows[index]["writer_id"] == "000"
        for index in selected
    )
    assert "heldout" not in identity["diagnostic_texts"]
    with pytest.raises(ValueError, match="digest"):
        load_scope(prepared, expected_records="b" * 64)
    with pytest.raises(ValueError, match="32 distinct"):
        load_scope(prepared, writer_id="001")
    (prepared / "records.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="records changed"):
        load_scope(prepared)


def test_review_requires_complete_human_answers_and_never_claims_generalization(tmp_path):
    model = small_model()
    items = [
        {
            "id": f"fake-{i}",
            "text": chr(97 + i // 26) + chr(97 + i % 26),
            "image": torch.zeros(1, 64, 48),
        }
        for i in range(32)
    ]
    output = tmp_path / "review"
    export_fit_review(model, items, output, seed=12, checkpoint_sha256="a" * 64)
    with pytest.raises(ValueError, match="Fill every"):
        score_review(output)
    with (output / "answer-key.csv").open(newline="", encoding="utf-8") as stream:
        key = list(csv.DictReader(stream))
    # Synthetic review fixture: copying labels here tests arithmetic, not model quality.
    rows = [
        {
            "sample_id": row["sample_id"],
            "transcription": row["requested"],
            "no_visual_defects": "yes",
        }
        for row in key
    ]
    rows[0]["transcription"] = "<unreadable>"
    write_csv(output / "transcriptions.csv", rows, list(rows[0]))
    scored = score_review(output)
    assert scored["exact_words"] == 31 and scored["fit_gate_passed"]
    assert not scored["generalization_established"]
    rows[1]["no_visual_defects"] = "no"
    write_csv(output / "transcriptions.csv", rows, list(rows[0]))
    assert not score_review(output)["fit_gate_passed"]
    (output / "sample-001.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        score_review(output)


def test_preflight_cannot_construct_an_optimizer(tmp_path, monkeypatch):
    prepared = make_prepared(tmp_path)
    config = FitConfig(
        prepared=str(prepared),
        cpu_threads=2,
        records_sha256=sha256_file(prepared / "records.jsonl"),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must not construct an optimizer")

    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    report = preflight(config, tmp_path / "preflight")
    assert report["weights_unchanged"] and not report["optimizer_constructed"]
    assert not report["training_performed"]
    assert not list((tmp_path / "preflight").glob("*.pt"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_updates": 1001},
        {"max_seconds": 1201},
        {"batch_size": 3},
        {"learning_rate": float("nan")},
    ],
)
def test_fit_budget_cannot_silently_expand(overrides):
    with pytest.raises(ValueError):
        FitConfig(**overrides)


def test_coordinate_migration_preserves_predictions_and_adds_trainable_position_inputs():
    model = small_model().eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    expanded = adapt_generator(model, "coordinate")
    texts, widths = ["ab", "letters"], torch.tensor([48, 160])
    original, migrated = model(texts, widths), expanded(texts, widths)
    torch.testing.assert_close(original["logits"], migrated["logits"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(original["predicted_widths"], migrated["predicted_widths"])
    alone = expanded(["ab"], widths[:1])
    torch.testing.assert_close(
        alone["logits"], migrated["logits"][:1, :, :, :48], atol=1e-6, rtol=1e-5
    )
    target = torch.zeros_like(migrated["logits"])
    target[:, :, 16:48, 14:20] = 1
    paired_loss(migrated, target, widths)["total"].backward()
    for old_block, block in zip(model.blocks, expanded.blocks, strict=True):
        old_channels = old_block[0].in_channels
        assert torch.count_nonzero(block[0].weight[:, old_channels:]) == 0
        gradient = block[0].weight.grad[:, old_channels:]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    with pytest.raises(ValueError, match="Only plain-to-coordinate"):
        adapt_generator(expanded, "plain")


def test_coordinate_checkpoint_roundtrip_and_legacy_config_load(tmp_path):
    provenance = {"writer_id": "000", "records_sha256": "a" * 64}
    path = tmp_path / "model.pt"
    original = small_model()
    save_generator(path, original, provenance)
    legacy = torch.load(path, weights_only=True)
    del legacy["model_config"]["architecture"]
    torch.save(legacy, path)
    loaded, _ = load_generator(path)
    assert loaded.config.architecture == "plain"
    coordinate = adapt_generator(loaded, "coordinate")
    save_generator(path, coordinate, provenance)
    again, _ = load_generator(path)
    torch.testing.assert_close(coordinate(["test"])["logits"], again(["test"])["logits"])


def test_warm_start_rejects_checkpoint_digest_and_training_identity_changes(tmp_path):
    prepared = make_prepared(tmp_path)
    _, _, _, identity = load_scope(prepared)
    path = tmp_path / "parent.pt"
    provenance = {
        "writer_id": "000",
        "records_sha256": identity["records_sha256"],
        "training_ids": identity["diagnostic_ids"],
    }
    parent = small_model()
    save_generator(path, parent, provenance)
    config = FitConfig(
        prepared=str(prepared),
        records_sha256=identity["records_sha256"],
        init_checkpoint=str(path),
        init_sha256=sha256_file(path),
        architecture="coordinate",
        cpu_threads=2,
    )
    migrated, _, _ = prepare(config)
    torch.testing.assert_close(
        parent(["test"])["logits"], migrated(["test"])["logits"], atol=1e-6, rtol=1e-5
    )
    altered_config = asdict(config)
    altered_config["init_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="pinned SHA256"):
        prepare(FitConfig(**altered_config))
    provenance["training_ids"] = identity["diagnostic_ids"][::-1]
    save_generator(path, parent, provenance)
    altered_config["init_sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="same writer, data and 32"):
        prepare(FitConfig(**altered_config))


@pytest.mark.parametrize(
    "overrides",
    [
        {"architecture": "unknown"},
        {"init_checkpoint": "parent.pt"},
        {"init_sha256": "a" * 64},
        {"init_checkpoint": "parent.pt", "init_sha256": "bad"},
    ],
)
def test_warm_start_requires_explicit_compatible_configuration(overrides):
    with pytest.raises(ValueError):
        FitConfig(**overrides)
