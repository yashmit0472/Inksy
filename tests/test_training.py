import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.model import ConditionalVAE, ModelConfig
from ml.train import TrainingConfig, run_epoch


@pytest.mark.training
def test_train_and_validation_account_for_partial_batches_and_keep_eval_deterministic():
    torch.manual_seed(3)
    model = ConditionalVAE(ModelConfig(latent_dim=4, label_dim=3, base_channels=4))
    data = TensorDataset(torch.rand(5, 1, 28, 28), torch.tensor([0, 1, 2, 3, 4]))
    loader = DataLoader(data, batch_size=3)
    opt = torch.optim.Adam(model.parameters())
    train = run_epoch(model, loader, torch.device("cpu"), 1, optimizer=opt)
    assert train["examples"] == 5
    first = run_epoch(model, loader, torch.device("cpu"), 1, seed=43)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    second = run_epoch(model, loader, torch.device("cpu"), 1, seed=43)
    assert first["loss"] == second["loss"]
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())
    assert run_epoch(model, loader, torch.device("cpu"), 1, max_batches=1)["examples"] == 3


@pytest.mark.parametrize(
    "values", [{"epochs": 0}, {"batch_size": 0}, {"beta": -1}, {"max_train_batches": 0}]
)
def test_invalid_training_config_is_rejected(values):
    with pytest.raises(ValueError):
        TrainingConfig(**values)


def test_validation_progress_uses_capped_total_and_does_not_update_weights(capsys):
    model = ConditionalVAE(ModelConfig(latent_dim=4, label_dim=3, base_channels=4))
    state = {key: value.clone() for key, value in model.state_dict().items()}
    dataset = TensorDataset(torch.rand(5, 1, 28, 28), torch.tensor([0, 1, 2, 3, 4]))
    metrics = run_epoch(
        model,
        DataLoader(dataset, batch_size=2),
        torch.device("cpu"),
        1,
        max_batches=2,
        description="Validation 1/3",
    )
    display = capsys.readouterr().err
    assert "Validation 1/3" in display
    assert "100%" in display and "2/2" in display
    assert "loss=" in display and "avg=" in display and "examples=4" in display
    assert metrics["examples"] == 4
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())


def test_progress_total_does_not_exceed_available_batches(capsys):
    model = ConditionalVAE(ModelConfig(latent_dim=4, label_dim=3, base_channels=4))
    dataset = TensorDataset(torch.rand(3, 1, 28, 28), torch.tensor([0, 1, 2]))
    metrics = run_epoch(
        model,
        DataLoader(dataset, batch_size=2),
        torch.device("cpu"),
        1,
        max_batches=100,
    )
    display = capsys.readouterr().err
    assert "100%" in display and "2/2" in display
    assert "examples=3" in display
    assert metrics["examples"] == 3


def test_progress_can_be_disabled_for_redirected_logs(capsys):
    model = ConditionalVAE(ModelConfig(latent_dim=4, label_dim=3, base_channels=4))
    dataset = TensorDataset(torch.rand(2, 1, 28, 28), torch.tensor([0, 1]))
    run_epoch(
        model,
        DataLoader(dataset, batch_size=2),
        torch.device("cpu"),
        1,
        show_progress=False,
    )
    assert capsys.readouterr().err == ""


def test_failed_batch_does_not_claim_completion(capsys):
    model = ConditionalVAE(ModelConfig(latent_dim=4, label_dim=3, base_channels=4))
    dataset = TensorDataset(torch.full((2, 1, 28, 28), float("nan")), torch.tensor([0, 1]))
    with pytest.raises(FloatingPointError, match="Non-finite"):
        run_epoch(model, DataLoader(dataset, batch_size=2), torch.device("cpu"), 1)
    display = capsys.readouterr().err
    assert "0/1" in display
    assert "100%" not in display
