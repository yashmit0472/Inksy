from itertools import islice

import pytest
import torch
from test_word_reader import fixture_items, tiny_reader
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ml.iam import collate_words
from ml.train_word_reader import CoverageSampler, ReaderTrainingConfig
from ml.word_reader import WordReaderConfig, adapt_reader, load_reader, save_reader, word_ctc_loss


def test_spatial_migration_preserves_predictions_and_original_parameters():
    original = tiny_reader().eval()
    state = {key: value.clone() for key, value in original.state_dict().items()}
    upgraded = adapt_reader(original, WordReaderConfig(2, 4, 1, "spatial"))
    batch = collate_words(fixture_items())
    with torch.inference_mode():
        before, lengths = original(batch["images"], batch["widths"])
        after, _ = upgraded(batch["images"], batch["widths"])
    assert torch.equal(before, after)
    assert all(torch.equal(value, state[key]) for key, value in original.state_dict().items())
    loss = word_ctc_loss(upgraded(batch["images"], batch["widths"])[0], lengths, batch)
    (gradient,) = torch.autograd.grad(loss, upgraded.height_projection.weight)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    # Gradient inspection alone must not fit the adapter.
    assert torch.count_nonzero(upgraded.height_projection.weight) == 0


def test_spatial_reader_preserves_padding_invariance():
    model = adapt_reader(tiny_reader().eval(), WordReaderConfig(2, 4, 1, "spatial"))
    batch = collate_words(fixture_items())
    with torch.inference_mode():
        actual, _ = model(batch["images"], batch["widths"])
        padded, _ = model(F.pad(batch["images"], (0, 32)), batch["widths"])
        alone, _ = model(batch["images"][:1, :, :, :32], torch.tensor([32]))
    assert torch.allclose(actual, padded[: len(actual)], atol=1e-6)
    assert torch.allclose(actual[:8, :1], alone, atol=1e-6)


def test_spatial_features_offer_distinct_learning_signals_for_vertical_positions():
    model = adapt_reader(tiny_reader(), WordReaderConfig(2, 4, 1, "spatial"))
    top = torch.zeros(1, 8, 16, 1)
    top[0, 0, 2, 0] = 1
    bottom = top.flip(2)
    assert torch.equal(top.mean(2), bottom.mean(2))
    (top_gradient,) = torch.autograd.grad(
        model.column_features(top).sum(), model.height_projection.weight
    )
    (bottom_gradient,) = torch.autograd.grad(
        model.column_features(bottom).sum(), model.height_projection.weight
    )
    assert not torch.equal(top_gradient, bottom_gradient)
    assert torch.count_nonzero(model.height_projection.weight) == 0


def test_old_checkpoint_default_and_spatial_round_trip_are_explicit(tmp_path):
    path = tmp_path / "reader.pt"
    save_reader(path, tiny_reader())
    payload = torch.load(path, weights_only=True)
    del payload["model_config"]["architecture"]
    torch.save(payload, path)
    legacy, _ = load_reader(path)
    assert legacy.config.architecture == "mean"
    spatial = adapt_reader(legacy, WordReaderConfig(2, 4, 1, "spatial"))
    save_reader(path, spatial)
    restored, _ = load_reader(path)
    assert restored.config.architecture == "spatial"
    assert all(
        torch.equal(value, restored.state_dict()[key])
        for key, value in spatial.state_dict().items()
    )
    with pytest.raises(ValueError, match="migration"):
        adapt_reader(spatial, WordReaderConfig(2, 4, 1, "mean"))
    with pytest.raises(ValueError, match="migration"):
        adapt_reader(legacy, WordReaderConfig(3, 4, 1, "spatial"))


def test_coverage_sampler_keeps_unused_examples_across_capped_loader_iterations():
    sampler = CoverageSampler(13, 314)
    loader = DataLoader(list(range(13)), batch_size=2, sampler=sampler)
    drawn = []
    for _ in range(5):
        for batch in islice(loader, 2):
            drawn.extend(batch.tolist())
    assert len(drawn) == 20 and len(set(drawn[:13])) == 13
    assert len(set(drawn[13:])) == 7
    assert sampler.report() == {
        "mode": "coverage",
        "draws": 20,
        "completed_data_passes": 1,
        "examples_into_next_pass": 7,
    }
    other = CoverageSampler(13, 314)
    assert list(other) == drawn[:13]


def test_spatial_preset_uses_latest_weights_and_bounded_coverage():
    import json
    from pathlib import Path

    config = ReaderTrainingConfig(
        **json.loads(Path("configs/word-reader-spatial-2500.json").read_text())
    )
    assert config.sampling == "coverage" and config.architecture == "spatial"
    assert config.init_checkpoint == "artifacts/word-reader-broad-2500/best.pt"
    assert config.epochs * config.max_train_batches == 2500
