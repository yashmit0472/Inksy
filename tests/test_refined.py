import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from ml.checkpoint import initialize_from_checkpoint, load_checkpoint, save_checkpoint
from ml.compare_generators import summarize_comparison
from ml.generate import sample_glyphs
from ml.model import ConditionalVAE, ModelConfig, vae_loss
from ml.train import TrainingConfig
from ml.utils import sha256_file


def make_model(architecture="baseline", latent_dim=4):
    return ConditionalVAE(
        ModelConfig(latent_dim=latent_dim, label_dim=3, base_channels=4, architecture=architecture)
    )


def test_refined_warm_start_preserves_baseline_pixels_and_source_checkpoint(tmp_path):
    baseline = make_model().eval()
    source = tmp_path / "baseline.pt"
    save_checkpoint(source, baseline, split_sha256="fixture", epoch=10)
    original_hash = sha256_file(source)
    refined = make_model("refined").eval()
    initialization = initialize_from_checkpoint(refined, source, "fixture")
    assert torch.equal(
        sample_glyphs(baseline, "Aa0Qth", 42, 2), sample_glyphs(refined, "Aa0Qth", 42, 2)
    )
    assert initialization["source_epoch"] == 10
    assert initialization["optimizer_restored"] is False
    assert initialization["new_parameters"]
    assert sha256_file(source) == original_hash


def test_refined_parameters_receive_gradients_without_optimizer_updates():
    model = make_model("refined")
    state = {key: value.clone() for key, value in model.state_dict().items()}
    images = torch.rand(3, 1, 28, 28)
    logits, mu, logvar = model(images, torch.tensor([0, 10, 61]))
    losses = vae_loss(logits, images, mu, logvar, edge_weight=0.5)
    losses["loss"].backward()
    assert model.condition_7.affine.weight.grad.abs().sum() > 0
    assert model.condition_14.affine.weight.grad.abs().sum() > 0
    assert model.stroke_refiner[-1].weight.grad.abs().sum() > 0
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())


def test_legacy_v1_checkpoint_still_loads_without_architecture_field(tmp_path):
    model = make_model().eval()
    source = tmp_path / "legacy.pt"
    save_checkpoint(source, model)
    payload = torch.load(source, weights_only=True)
    payload["format_version"] = 1
    payload["model_config"].pop("architecture")
    torch.save(payload, source)
    restored, _ = load_checkpoint(source)
    assert restored.config.architecture == "baseline"
    assert torch.equal(sample_glyphs(model, "Ab9", 123), sample_glyphs(restored, "Ab9", 123))


def test_refined_checkpoint_roundtrip_is_exact(tmp_path):
    model = make_model("refined").eval()
    path = tmp_path / "refined.pt"
    save_checkpoint(path, model)
    restored, metadata = load_checkpoint(path)
    assert metadata["format_version"] == 2
    assert restored.config.architecture == "refined"
    assert torch.equal(sample_glyphs(model, "AaQ", 2), sample_glyphs(restored, "AaQ", 2))


@pytest.mark.parametrize("mismatch", ["split", "dimension", "downgrade"])
def test_unsafe_initialization_is_rejected(tmp_path, mismatch):
    source_model = make_model("refined" if mismatch == "downgrade" else "baseline")
    path = tmp_path / "source.pt"
    save_checkpoint(path, source_model, split_sha256="fixture")
    target = make_model(
        "baseline" if mismatch == "downgrade" else "refined",
        latent_dim=5 if mismatch == "dimension" else 4,
    )
    with pytest.raises(ValueError):
        initialize_from_checkpoint(target, path, "other" if mismatch == "split" else "fixture")


def test_edge_objective_penalizes_smoothed_strokes_and_preserves_common_base_loss():
    target = torch.zeros(2, 1, 28, 28)
    target[:, :, 5:23, 12:16] = 1
    sharp = torch.logit(target.clamp(0.001, 0.999)).requires_grad_()
    soft = torch.logit(F.avg_pool2d(target, 3, 1, 1).clamp(0.001, 0.999))
    stats = torch.zeros(2, 4)
    sharp_loss = vae_loss(sharp, target, stats, stats, edge_weight=0.5)
    soft_loss = vae_loss(soft, target, stats, stats, edge_weight=0.5)
    base = vae_loss(sharp, target, stats, stats)
    assert soft_loss["edge"] > sharp_loss["edge"]
    assert torch.equal(base["loss"], sharp_loss["base_loss"])
    assert sharp_loss["loss"].item() == pytest.approx(
        sharp_loss["base_loss"].item() + 0.5 * sharp_loss["edge"].item()
    )
    sharp_loss["edge"].backward()
    assert torch.isfinite(sharp.grad).all() and sharp.grad.abs().sum() > 0


@pytest.mark.parametrize("values", [{"architecture": "unknown"}, {"edge_weight": -1}])
def test_invalid_refinement_config_is_rejected(values):
    with pytest.raises(ValueError):
        TrainingConfig(**values)


def test_refinement_and_continuation_control_share_experiment_settings():
    root = Path(__file__).resolve().parents[1]
    refined = json.loads((root / "configs/refined.json").read_text())
    control = json.loads((root / "configs/continue-baseline.json").read_text())
    expected_changes = {"architecture", "edge_weight", "output_dir"}
    assert {key for key in refined if refined[key] != control[key]} == expected_changes
    assert refined["init_checkpoint"] == "artifacts/baseline-10/best.pt"
    assert refined["kl_warmup_epochs"] == 0


def test_comparison_reports_deltas_without_certifying_quality():
    def fixture(loss, accuracy, edge):
        return {
            "validation": {"metrics": {"loss": loss, "edge": edge}},
            "agreement": {
                "generated_agreement": {"accuracy": accuracy, "case_insensitive_accuracy": accuracy}
            },
            "quality": {
                "generated": {
                    "soft_ink_fraction": 0.8,
                    "mean_edge_energy": 0.02,
                    "near_blank_fraction": 0.0,
                }
            },
        }

    report = summarize_comparison(fixture(130, 0.6, 20), fixture(128, 0.7, 18))
    assert report["candidate_minus_baseline"]["common_validation_loss"] == -2
    assert report["candidate_minus_baseline"]["exact_label_agreement"] == pytest.approx(0.1)
    assert report["release_gate_passed"] is False
