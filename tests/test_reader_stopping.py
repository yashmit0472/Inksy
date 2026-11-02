import json
from pathlib import Path

import pytest

from ml.train_word_reader import CERStopper, ReaderTrainingConfig


def test_plateau_stopping_counts_from_initial_selection_and_accepts_cumulative_gains():
    monitor = CERStopper(patience=3, min_delta=0.005)
    assert not monitor.observe(0.40)
    assert not monitor.observe(0.398)
    assert not monitor.observe(0.396)
    assert not monitor.observe(0.392)  # Cumulative gain clears the threshold.
    assert monitor.stale_checks == 0
    assert not monitor.observe(0.393)
    assert not monitor.observe(0.392)
    assert monitor.observe(0.391)


def test_plateau_control_disabled_for_existing_configs_and_zero_delta_requires_improvement():
    disabled = CERStopper(0, 0.005)
    assert not any(disabled.observe(0.4) for _ in range(20))
    enabled = CERStopper(2, 0.0)
    assert not enabled.observe(0.4)
    assert not enabled.observe(0.4)
    assert enabled.observe(0.4)
    assert not enabled.observe(0.399)


@pytest.mark.parametrize(
    "values",
    [
        {"early_stopping_patience": -1},
        {"early_stopping_patience": True},
        {"early_stopping_min_delta": float("nan")},
        {"early_stopping_min_delta": -0.1},
    ],
)
def test_invalid_stopping_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        ReaderTrainingConfig(**values)


def test_continuation_keeps_spatial_model_and_latest_own_weights_with_finite_budget():
    config = ReaderTrainingConfig(
        **json.loads(Path("configs/word-reader-spatial-continue.json").read_text())
    )
    assert config.architecture == "spatial" and config.sampling == "coverage"
    assert config.init_checkpoint == "artifacts/word-reader-spatial-2500/best.pt"
    assert config.epochs * config.max_train_batches == 5000
    assert config.early_stopping_patience == 3
    assert config.learning_rate == 0.0005
