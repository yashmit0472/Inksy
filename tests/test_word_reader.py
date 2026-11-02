import json

import pytest
import torch
from test_iam import iam_fixture as iam_fixture
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ml.checkpoint import load_checkpoint
from ml.iam import WORD_ALPHABET, WORD_TOKENS, collate_words, encode_word, prepare
from ml.train_word_reader import ReaderTrainingConfig, evaluation_loader, reader_epoch
from ml.word_reader import (
    WordReader,
    WordReaderConfig,
    edit_distance,
    greedy_decode,
    load_reader,
    reading_metrics,
    save_reader,
    word_ctc_loss,
)


def tiny_reader():
    return WordReader(WordReaderConfig(channels=2, hidden=4, layers=1))


def fixture_items():
    return [
        {
            "image": torch.rand(1, 64, width),
            "text": text,
            "tokens": encode_word(text),
            "writer_index": 0,
            "id": str(index),
        }
        for index, (width, text) in enumerate(((32, "aa"), (40, "Book"), (48, "9")))
    ]


def test_word_reader_is_invariant_to_right_padding_and_other_batch_members():
    model = tiny_reader().eval()
    first, other, _ = fixture_items()
    with torch.inference_mode():
        alone, alone_lengths = model(first["image"].unsqueeze(0), torch.tensor([32]))
        batch = collate_words([first, other])
        together, lengths = model(batch["images"], batch["widths"])
        extra_padding, _ = model(F.pad(first["image"].unsqueeze(0), (0, 32)), torch.tensor([32]))
    assert alone.shape == (8, 1, len(WORD_ALPHABET) + 1)
    assert torch.allclose(alone[:, 0], together[:8, 0], atol=1e-6)
    assert torch.allclose(alone, extra_padding[:8], atol=1e-6)
    assert alone_lengths.tolist() == [8] and lengths.tolist() == [8, 10]


def test_frozen_word_reader_passes_ctc_gradients_to_images_without_updating_weights():
    model = tiny_reader().eval().requires_grad_(False)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    batch = collate_words(fixture_items())
    images = batch["images"].clone().requires_grad_()
    log_probabilities, lengths = model(images, batch["widths"])
    loss = word_ctc_loss(log_probabilities, lengths, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(images.grad).all() and images.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())


def test_greedy_ctc_decoding_preserves_repeated_letters_separated_by_blanks():
    tokens = WORD_TOKENS
    path = [
        tokens["b"],
        tokens["b"],
        0,
        tokens["o"],
        tokens["o"],
        0,
        tokens["o"],
        tokens["k"],
        tokens["k"],
        tokens["z"],
        tokens["z"],
    ]
    probabilities = F.one_hot(torch.tensor(path), len(WORD_ALPHABET) + 1).float()[:, None, :]
    assert greedy_decode(probabilities, torch.tensor([9])) == ["book"]
    assert greedy_decode(torch.zeros_like(probabilities), torch.tensor([9])) == [""]


def test_character_error_rate_counts_insertions_and_is_not_clamped_to_one():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("aa", "") == 2
    metrics = reading_metrics([{"expected": "A", "predicted": "BBBB"}])
    assert metrics["character_error_rate"] == 4
    metrics = reading_metrics(
        [{"expected": "Book", "predicted": "book"}, {"expected": "cat", "predicted": "cat"}]
    )
    assert metrics["exact_word_accuracy"] == 0.5
    assert metrics["case_insensitive_word_accuracy"] == 1
    assert metrics["character_error_rate"] == pytest.approx(1 / 7)


def test_ctc_rejects_impossible_alignment_and_blank_targets_instead_of_hiding_errors():
    batch = collate_words(fixture_items())
    probabilities = torch.randn(10, 3, len(WORD_ALPHABET) + 1).log_softmax(-1)
    with pytest.raises(ValueError, match="align"):
        word_ctc_loss(probabilities, torch.tensor([2, 10, 10]), batch)
    batch["ctc_targets"][0] = 0
    with pytest.raises(ValueError, match="blank"):
        word_ctc_loss(probabilities, torch.tensor([10, 10, 10]), batch)


def test_word_reader_checkpoint_is_separate_from_glyph_generator_contract(tmp_path):
    model = tiny_reader().eval()
    path = tmp_path / "reader.pt"
    save_reader(path, model, epoch=0, records_sha256="fixture", quality_status="test-fixture")
    restored, metadata = load_reader(path)
    batch = collate_words(fixture_items())
    with torch.inference_mode():
        expected, _ = model(batch["images"], batch["widths"])
        observed, _ = restored(batch["images"], batch["widths"])
    assert torch.equal(expected, observed)
    assert metadata["role"] == "training-content-reader"
    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    payload["role"] = "independent-generator-evaluator"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="role"):
        load_reader(path)


def test_validation_reports_actual_examples_and_does_not_train(capsys):
    model = tiny_reader()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    loader = DataLoader(fixture_items(), batch_size=2, collate_fn=collate_words)
    first, rows = reader_epoch(model, loader, torch.device("cpu"))
    second, _ = reader_epoch(model, loader, torch.device("cpu"), show_progress=False)
    assert first["examples"] == 3 and len(rows) == 3
    assert first["optimizer_updates"] == 0
    assert first["loss"] == second["loss"]
    assert first["character_error_rate"] == second["character_error_rate"]
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    display = capsys.readouterr().err
    assert "2/2" in display and "100%" in display and "CER=" in display


def test_evaluation_cap_samples_across_partition_and_keeps_identity_fixed(request, tmp_path):
    prepared = tmp_path / "prepared"
    prepare(request.getfixturevalue("iam_fixture"), prepared)
    loader, identity = evaluation_loader(prepared, "content_validation", 2, max_batches=1)
    _, repeated = evaluation_loader(prepared, "content_validation", 2, max_batches=1)
    assert identity == repeated
    assert identity["evaluated_examples"] == 2 and identity["partial"]
    batch = next(iter(loader))
    assert len(batch["text"]) == 2
    with pytest.raises(ValueError, match="development"):
        evaluation_loader(prepared, "test", 2)


@pytest.mark.parametrize(
    "values",
    [
        {"epochs": 0},
        {"max_train_batches": 0},
        {"max_validation_batches": -1},
        {"batch_size": True},
        {"learning_rate": float("nan")},
        {"seed": -1},
        {"layers": 0},
    ],
)
def test_invalid_word_reader_configs_are_rejected(values):
    with pytest.raises(ValueError):
        ReaderTrainingConfig(**values)


@pytest.mark.training
def test_owner_opt_in_word_reader_epoch_updates_weights_on_synthetic_fixtures():
    model = tiny_reader()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    loader = DataLoader(fixture_items(), batch_size=2, collate_fn=collate_words)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics, _ = reader_epoch(
        model, loader, torch.device("cpu"), optimizer=optimizer, show_progress=False
    )
    assert metrics["optimizer_updates"] == 2 and metrics["examples"] == 3
    assert any(not torch.equal(value, state[key]) for key, value in model.state_dict().items())


def test_pilot_budget_is_explicit_and_uses_the_prepared_word_data():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs/word-reader-pilot.json"
    config = ReaderTrainingConfig(**json.loads(path.read_text()))
    assert config.epochs * config.max_train_batches == 2500
    assert config.epochs * config.max_train_batches * config.batch_size == 40_000
    assert config.prepared == "artifacts/iam-words" and config.device == "cpu"
