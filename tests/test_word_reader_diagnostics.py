import json

import pytest
import torch
from test_iam import iam_fixture as iam_fixture
from test_word_reader import fixture_items, tiny_reader

from ml.diagnose_word_reader import ablate_images, diagnose
from ml.iam import collate_words, prepare
from ml.train_word_reader_sanity import SanityConfig, passes_memorization, sanity_batch
from ml.utils import sha256_file, write_json


def test_ablations_keep_actual_widths_and_do_not_reverse_padding_into_words():
    batch = collate_words(fixture_items())
    original = batch["images"].clone()
    reversed_images = ablate_images(original, batch["widths"], "reversed")
    assert torch.equal(reversed_images[0, :, :, :32], original[0, :, :, :32].flip(-1))
    assert torch.count_nonzero(reversed_images[0, :, :, 32:]) == 0
    assert torch.equal(ablate_images(reversed_images, batch["widths"], "reversed"), original)
    assert not torch.count_nonzero(ablate_images(original, batch["widths"], "blank"))
    assert torch.equal(batch["images"], original)


def test_diagnostics_leave_parameters_unchanged_and_measure_only_valid_frames():
    model = tiny_reader()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    batch = collate_words(fixture_items())
    metrics, rows = diagnose(model, [batch])
    assert metrics["real"]["examples"] == 3 and len(rows["blank"]) == 3
    for name in ("blank", "reversed"):
        assert 0 <= metrics[name]["mean_frame_probability_total_variation"] <= 1
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    padded = {**batch, "images": torch.nn.functional.pad(batch["images"], (0, 32))}
    repeated, _ = diagnose(model, [padded])
    for name in metrics:
        assert metrics[name]["ctc_loss"] == pytest.approx(repeated[name]["ctc_loss"], abs=1e-6)
    assert metrics["blank"]["mean_frame_probability_total_variation"] == pytest.approx(
        repeated["blank"]["mean_frame_probability_total_variation"], abs=1e-6
    )


def test_memorization_gate_rejects_width_only_success_and_incomplete_learning():
    metrics = {
        "real": {"exact_word_accuracy": 1.0, "ctc_loss": 0.1},
        "blank": {"exact_word_accuracy": 0.0, "ctc_loss": 4.0},
    }
    assert passes_memorization(metrics)
    metrics["blank"]["exact_word_accuracy"] = 1.0
    assert not passes_memorization(metrics)
    metrics["blank"]["exact_word_accuracy"] = 0.0
    metrics["real"]["exact_word_accuracy"] = 15 / 16
    assert not passes_memorization(metrics)


def test_sanity_subset_verifies_hash_and_refuses_nontraining_ids(request, tmp_path):
    prepared = tmp_path / "prepared"
    prepare(request.getfixturevalue("iam_fixture"), prepared)
    batch, identity = sanity_batch(prepared)
    assert len(batch["images"]) == 16 and identity["partition"] == "train"
    path = prepared / "sanity-set.json"
    specification = json.loads(path.read_text())
    specification["word_ids"][0] = "a01-001u-00-00"  # Reserved writer-validation image.
    write_json(path, specification)
    with pytest.raises(ValueError, match="identity changed"):
        sanity_batch(prepared)
    manifest = json.loads((prepared / "manifest.json").read_text())
    manifest["sanity_set_sha256"] = sha256_file(path)
    write_json(prepared / "manifest.json", manifest)
    with pytest.raises(ValueError, match="outside the training partition"):
        sanity_batch(prepared)


@pytest.mark.parametrize(
    "values",
    [
        {"max_updates": 2001},
        {"max_updates": 0},
        {"check_every": 601},
        {"learning_rate": float("nan")},
        {"consecutive_passes": False},
    ],
)
def test_memorization_config_rejects_invalid_or_unbounded_runs(values):
    with pytest.raises(ValueError):
        SanityConfig(**values)
