import json
from pathlib import Path

import pytest
import torch
from test_word_reader import fixture_items, tiny_reader
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ml.iam import collate_words
from ml.reader_balance import FrequencyBalancedCollator
from ml.train_word_reader import ReaderTrainingConfig, reader_epoch
from ml.word_reader import word_ctc_loss


def test_training_only_counts_casefold_variants_and_bounds_rare_word_weight():
    rows = [{"text": "aa", "partition": "train"}] * 400
    rows += [
        {"text": "AA", "partition": "train"},
        {"text": "Book", "partition": "train"},
        {"text": "9", "partition": "train"},
    ]
    collator = FrequencyBalancedCollator(rows)
    items = fixture_items()
    batch = collator(items)
    assert batch["reader_loss_weights"].tolist() == pytest.approx([0.1, 1, 1])
    plain = collate_words(items)
    for name in ("images", "widths", "ctc_targets", "token_lengths"):
        assert torch.equal(batch[name], plain[name])
    assert batch["text"] == plain["text"] and batch["ids"] == plain["ids"]
    assert collator.report()["distinct_casefold_words"] == 3
    assert (
        collator.report()["frequency_sha256"]
        == FrequencyBalancedCollator(rows[::-1]).report()["frequency_sha256"]
    )
    with pytest.raises(ValueError, match="training records"):
        FrequencyBalancedCollator(rows + [{"text": "heldout", "partition": "content_validation"}])
    with pytest.raises(ValueError, match="absent"):
        FrequencyBalancedCollator(rows[:400])(items)


def test_weighted_ctc_matches_explicit_per_word_loss_and_has_finite_gradients():
    model = tiny_reader()
    batch = collate_words(fixture_items())
    logp, lengths = model(batch["images"], batch["widths"])
    weights = torch.tensor([0.1, 1.0, 0.5])
    loss = word_ctc_loss(logp, lengths, batch, sample_weights=weights)
    individual = F.ctc_loss(
        logp, batch["ctc_targets"], lengths, batch["token_lengths"], reduction="none"
    )
    expected = ((individual / batch["token_lengths"]) * weights).sum() / weights.sum()
    assert torch.allclose(loss, expected)
    assert torch.allclose(
        word_ctc_loss(logp, lengths, batch),
        word_ctc_loss(logp, lengths, batch, sample_weights=torch.ones(3)),
    )
    assert torch.allclose(loss, word_ctc_loss(logp, lengths, batch, sample_weights=weights * 3))
    state = {key: value.clone() for key, value in model.state_dict().items()}
    (gradient,) = torch.autograd.grad(loss, model.classifier.weight)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())


@pytest.mark.parametrize(
    "weights",
    [
        torch.tensor([0.0, 1.0, 1.0]),
        torch.tensor([-1.0, 1.0, 1.0]),
        torch.tensor([float("nan"), 1.0, 1.0]),
        torch.ones(2),
        torch.ones(3, 1),
    ],
)
def test_weighted_ctc_rejects_invalid_weights(weights):
    batch = collate_words(fixture_items())
    logp, lengths = tiny_reader()(batch["images"], batch["widths"])
    with pytest.raises(ValueError, match="sample weights"):
        word_ctc_loss(logp, lengths, batch, sample_weights=weights)


def test_validation_is_unweighted_even_if_a_batch_contains_training_weights():
    items = fixture_items()
    rows = [{"text": "aa", "partition": "train"}] * 100
    rows += [{"text": "Book", "partition": "train"}, {"text": "9", "partition": "train"}]
    weighted = DataLoader(items, batch_size=3, collate_fn=FrequencyBalancedCollator(rows))
    plain = DataLoader(items, batch_size=3, collate_fn=collate_words)
    model = tiny_reader()
    first, first_rows = reader_epoch(model, weighted, torch.device("cpu"), show_progress=False)
    second, second_rows = reader_epoch(model, plain, torch.device("cpu"), show_progress=False)
    assert first["loss"] == second["loss"] and first_rows == second_rows
    assert first["optimizer_updates"] == 0


def test_balanced_preset_preserves_latest_model_and_full_unweighted_evaluation():
    config = ReaderTrainingConfig(
        **json.loads(Path("configs/word-reader-balanced.json").read_text())
    )
    assert config.balance_word_frequency and config.architecture == "spatial"
    assert config.init_checkpoint == "artifacts/word-reader-spatial-continue-2/best.pt"
    assert config.max_validation_batches is None and config.sampling == "coverage"
    assert config.epochs * config.max_train_batches == 5000
    assert not ReaderTrainingConfig().balance_word_frequency
    with pytest.raises(ValueError, match="boolean"):
        ReaderTrainingConfig(balance_word_frequency="true")
