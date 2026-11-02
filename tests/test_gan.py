import importlib
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.checkpoint import initialize_from_checkpoint, load_checkpoint, save_checkpoint
from ml.compare_gan import summarize_comparison
from ml.evaluate import evaluate
from ml.gan import (
    ConditionalGenerator,
    GANModelConfig,
    ProjectionDiscriminator,
    discriminator_loss,
    generator_loss,
)
from ml.generate import sample_glyphs
from ml.model import ConditionalVAE, ModelConfig
from ml.preview import create_preview
from ml.train_gan import GANTrainingConfig, frozen_discriminator, run_gan_epoch
from ml.utils import sha256_file
from rendering.compositor import LayoutOptions


def models():
    config = GANModelConfig(latent_dim=4, label_dim=3, base_channels=4)
    return ConditionalGenerator(config), ProjectionDiscriminator(config)


def snapshot(model):
    return {key: value.clone() for key, value in model.state_dict().items()}


def assert_unchanged(model, state):
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())


def test_gan_sampling_is_deterministic_conditioned_and_bounded_without_weight_updates():
    generator, discriminator = models()
    generator.eval()
    discriminator.eval()
    state = snapshot(generator)
    rng_state = torch.random.get_rng_state().clone()
    first = sample_glyphs(generator, "Aa09", 123, 2)
    second = sample_glyphs(generator, "Aa09", 123, 2)
    assert torch.equal(rng_state, torch.random.get_rng_state())
    assert torch.equal(first, second)
    assert first.shape == (8, 1, 28, 28)
    assert torch.isfinite(first).all() and first.min() >= 0 and first.max() <= 1
    assert not torch.equal(first, sample_glyphs(generator, "Aa09", 124, 2))
    noise = torch.zeros(2, generator.config.latent_dim)
    assert not torch.allclose(
        generator(noise, torch.tensor([0, 1]))[0], generator(noise, torch.tensor([0, 1]))[1]
    )
    scores = discriminator(first[:1].expand(2, -1, -1, -1), torch.tensor([0, 1]))
    assert scores.shape == (2,) and not torch.allclose(scores[0], scores[1])
    assert_unchanged(generator, state)
    assert all(parameter.grad is None for parameter in generator.parameters())


def test_discriminator_gradients_do_not_flow_into_detached_generator():
    generator, discriminator = models()
    # Evaluation mode freezes spectral-norm power-iteration buffers in this non-training check.
    discriminator.eval()
    g_state, d_state = snapshot(generator), snapshot(discriminator)
    labels = torch.tensor([0, 10, 61])
    with torch.no_grad():
        fake = generator(torch.randn(3, 4), labels)
    loss = discriminator_loss(
        discriminator(torch.rand_like(fake), labels), discriminator(fake, labels)
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in generator.parameters())
    assert any(parameter.grad.abs().sum() > 0 for parameter in discriminator.parameters())
    assert all(torch.isfinite(p.grad).all() for p in discriminator.parameters())
    assert_unchanged(generator, g_state)
    assert_unchanged(discriminator, d_state)


def test_generator_gradients_pass_through_frozen_discriminator_without_mutation():
    generator, discriminator = models()
    g_state, d_state = snapshot(generator), snapshot(discriminator)
    with frozen_discriminator(discriminator):
        loss = generator_loss(
            discriminator(
                generator(torch.randn(3, 4), torch.tensor([0, 10, 61])), torch.tensor([0, 10, 61])
            )
        )
        loss.backward()
    assert discriminator.training
    assert all(parameter.requires_grad for parameter in discriminator.parameters())
    assert all(parameter.grad is None for parameter in discriminator.parameters())
    for parameter in generator.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert generator.label_embedding.weight.grad.abs().sum() > 0
    assert generator.condition_14.affine.weight.grad.abs().sum() > 0
    assert generator.output[-1].weight.grad.abs().sum() > 0
    assert_unchanged(generator, g_state)
    assert_unchanged(discriminator, d_state)


def test_frozen_discriminator_restores_flags_even_after_an_error():
    _, discriminator = models()
    next(discriminator.parameters()).requires_grad_(False)
    flags = [p.requires_grad for p in discriminator.parameters()]
    discriminator.eval()
    with pytest.raises(RuntimeError, match="fixture"):
        with frozen_discriminator(discriminator):
            raise RuntimeError("fixture")
    assert not discriminator.training
    assert [p.requires_grad for p in discriminator.parameters()] == flags


def test_hinge_losses_have_expected_values_and_gradient_directions():
    real = torch.tensor([2.0, 0.5, -1.0], requires_grad=True)
    fake = torch.tensor([-2.0, 0.5, 2.0], requires_grad=True)
    loss = discriminator_loss(real, fake)
    assert float(loss.detach()) == pytest.approx(7 / 3)
    loss.backward()
    assert torch.allclose(real.grad, torch.tensor([0.0, -1 / 3, -1 / 3]))
    assert torch.allclose(fake.grad, torch.tensor([0.0, 1 / 3, 1 / 3]))
    assert float(generator_loss(fake).detach()) == pytest.approx(-1 / 6)


def test_cgan_checkpoint_roundtrip_and_existing_sampling_contract(tmp_path):
    generator, _ = models()
    generator.eval()
    path = tmp_path / "generator.pt"
    save_checkpoint(path, generator, epoch=0, split_sha256="fixture", quality_status="test-fixture")
    original_hash = sha256_file(path)
    restored, metadata = load_checkpoint(path)
    assert metadata["model_family"] == "cgan"
    assert isinstance(restored, ConditionalGenerator)
    assert torch.equal(sample_glyphs(generator, "AbQ9", 42), sample_glyphs(restored, "AbQ9", 42))
    preview = create_preview(
        "Hello 123!",
        checkpoint=path,
        output=tmp_path / "preview",
        options=LayoutOptions(width=500, height=500, margin=32),
        show_progress=False,
    )
    assert preview["training_performed"] is False
    assert preview["coverage"]["model_glyphs"] == 8
    assert sha256_file(path) == original_hash


def test_cgan_cannot_be_used_for_vae_initialization_or_reconstruction(tmp_path):
    generator, _ = models()
    path = tmp_path / "generator.pt"
    save_checkpoint(path, generator, split_sha256="fixture")
    vae = ConditionalVAE(ModelConfig(latent_dim=4, label_dim=3, base_channels=4))
    with pytest.raises(ValueError, match="VAE checkpoint"):
        initialize_from_checkpoint(vae, path, "fixture")
    with pytest.raises(ValueError, match="do not apply to cGANs"):
        evaluate(path, tmp_path / "unused", tmp_path / "unused", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_unknown_checkpoint_family_is_rejected(tmp_path):
    generator, _ = models()
    path = tmp_path / "bad.pt"
    save_checkpoint(path, generator)
    payload = torch.load(path, weights_only=True)
    payload["model_family"] = "unknown"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="model_family"):
        load_checkpoint(path)


@pytest.mark.parametrize(
    "values",
    [
        {"epochs": 0},
        {"max_train_batches": 0},
        {"max_train_batches": 1.5},
        {"batch_size": True},
        {"generator_lr": float("nan")},
        {"discriminator_lr": -1},
        {"adam_beta1": 1},
        {"adam_beta2": -1},
        {"seed": -1},
        {"device": "bad"},
    ],
)
def test_invalid_pilot_config_is_rejected(values):
    with pytest.raises(ValueError):
        GANTrainingConfig(**values)


def test_preset_is_bounded_and_starts_independently():
    path = Path(__file__).resolve().parents[1] / "configs/cgan-pilot.json"
    config = GANTrainingConfig(**json.loads(path.read_text()))
    assert config.epochs * config.max_train_batches == 1500
    assert config.epochs * config.max_train_batches * config.batch_size == 192_000
    assert config.device == "cpu" and config.progress


def test_comparison_does_not_invent_vae_loss_or_certify_quality():
    def bundle(accuracy):
        return {
            "agreement": {
                "generated_agreement": {
                    "accuracy": accuracy,
                    "case_insensitive_accuracy": accuracy,
                }
            },
            "quality": {
                "generated": {
                    "soft_ink_fraction": 0.5,
                    "mean_edge_energy": 0.02,
                    "near_blank_fraction": 0.0,
                }
            },
            "diversity": {"mean_within_class_pixel_l1": 0.1},
        }

    report = summarize_comparison(bundle(0.6), bundle(0.7))
    assert report["candidate_minus_baseline"]["exact_label_agreement"] == pytest.approx(0.1)
    assert report["vae_loss_comparison"] is None
    assert not report["release_gate_passed"] and not report["human_review_completed"]
    assert not report["training_performed"]


def test_diversity_diagnostic_identifies_exact_collapse_without_training(monkeypatch):
    module = importlib.import_module("ml.compare_gan")
    monkeypatch.setattr(module, "ALPHABET", "A")
    monkeypatch.setattr(module, "sample_glyphs", lambda *args: torch.zeros(4, 1, 28, 28))
    result = module.diversity_diagnostics(None, samples_per_class=4)
    assert result["mean_within_class_pixel_l1"] == 0
    assert result["per_class"]["A"]["distinct_8bit_images"] == 1


def test_comparison_exports_matching_passages_and_unfilled_review_rows(tmp_path, monkeypatch):
    """Fixture metrics exercise report wiring, not learned handwriting quality."""
    module = importlib.import_module("ml.compare_gan")
    baseline = ConditionalVAE(ModelConfig(latent_dim=4, label_dim=3, base_channels=4))
    candidate, _ = models()
    baseline_path, candidate_path = tmp_path / "vae.pt", tmp_path / "gan.pt"
    for path, model in ((baseline_path, baseline), (candidate_path, candidate)):
        save_checkpoint(path, model, split_sha256="fixture", quality_status="test-fixture", epoch=0)
    bundle = {
        "agreement": {"generated_agreement": {"accuracy": 0.5, "case_insensitive_accuracy": 0.6}},
        "quality": {
            "generated": {
                "soft_ink_fraction": 0.5,
                "mean_edge_energy": 0.02,
                "near_blank_fraction": 0.0,
            }
        },
        "diversity": {"mean_within_class_pixel_l1": 0.1},
    }
    monkeypatch.setattr(module, "score_generation", lambda *args: bundle["agreement"])
    monkeypatch.setattr(module, "diagnose", lambda *args, **kwargs: bundle["quality"])
    monkeypatch.setattr(module, "diversity_diagnostics", lambda *args: bundle["diversity"])
    passages = []
    monkeypatch.setattr(
        module,
        "create_preview",
        lambda text, **kwargs: passages.append((text, kwargs["seed"], kwargs["checkpoint"])),
    )
    recognizer = tmp_path / "recognizer.pt"
    recognizer.write_bytes(b"fixture identity only")
    output = tmp_path / "comparison"
    hashes = (sha256_file(baseline_path), sha256_file(candidate_path))
    report = module.compare(candidate_path, output, baseline=baseline_path, recognizer=recognizer)
    assert passages[0][:2] == passages[1][:2]
    assert passages[0][2] == baseline_path and passages[1][2] == candidate_path
    import csv

    with (output / "human-review.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 62 * 4 * 2
    assert all(row["observed_character"] == row["readable_yes_no"] == "" for row in rows)
    assert not report["training_performed"] and report["vae_loss_comparison"] is None
    assert hashes == (sha256_file(baseline_path), sha256_file(candidate_path))
    assert json.loads((output / "comparison.json").read_text())["human_review_completed"] is False


@pytest.mark.training
def test_owner_opt_in_gan_epoch_updates_both_models_and_accounts_for_partial_batch(capsys):
    generator, discriminator = models()
    g_state, d_state = snapshot(generator), snapshot(discriminator)
    dataset = TensorDataset(torch.rand(5, 1, 28, 28), torch.tensor([0, 1, 2, 3, 4]))
    metrics = run_gan_epoch(
        generator,
        discriminator,
        DataLoader(dataset, batch_size=3),
        torch.optim.Adam(generator.parameters()),
        torch.optim.Adam(discriminator.parameters()),
        torch.device("cpu"),
        torch.Generator().manual_seed(42),
        max_batches=2,
    )
    assert metrics["examples"] == 5
    assert metrics["generator_updates"] == metrics["discriminator_updates"] == 2
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(
        not torch.equal(value, g_state[key]) for key, value in generator.state_dict().items()
    )
    assert any(
        not torch.equal(value, d_state[key]) for key, value in discriminator.state_dict().items()
    )
    assert "2/2" in capsys.readouterr().err
