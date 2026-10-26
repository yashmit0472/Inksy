import math

import pytest
import torch

from ml.checkpoint import load_checkpoint, save_checkpoint
from ml.generate import sample_glyphs
from ml.model import ConditionalVAE, ModelConfig, vae_loss


def small_model():
    return ConditionalVAE(ModelConfig(latent_dim=4, label_dim=3, base_channels=4))


def test_forward_loss_and_gradients_reach_both_conditioned_networks():
    model = small_model()
    targets = torch.rand(3, 1, 28, 28)
    labels = torch.tensor([0, 10, 61])
    logits, mu, logvar = model(targets, labels)
    assert logits.shape == targets.shape
    assert mu.shape == logvar.shape == (3, 4)
    losses = vae_loss(logits, targets, mu, logvar)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.decoder[-1].weight.grad.abs().sum() > 0
    assert model.label_embedding.weight.grad[[0, 10, 61]].abs().sum() > 0


def test_loss_reductions_have_known_scale_and_zero_kl():
    logits = torch.zeros(2, 1, 28, 28)
    stats = torch.zeros(2, 4)
    loss = vae_loss(logits, torch.zeros_like(logits), stats, stats)
    assert loss["kl"].item() == 0
    assert loss["reconstruction"].item() == pytest.approx(784 * math.log(2))


def test_requested_label_is_used_by_decoder():
    model = small_model().eval()
    latent = torch.zeros(2, 4)
    with torch.no_grad():
        output = model.decode(latent, torch.tensor([10, 36]))
    assert not torch.equal(output[0], output[1])


def test_sampling_is_reproducible_and_does_not_change_global_rng():
    model = small_model().eval()
    state = torch.get_rng_state()
    first = sample_glyphs(model, "Aa9", 123, 2)
    second = sample_glyphs(model, "Aa9", 123, 2)
    assert torch.equal(first, second)
    assert torch.equal(state, torch.get_rng_state())
    assert not torch.equal(first, sample_glyphs(model, "Aa9", 124, 2))


def test_checkpoint_roundtrip_preserves_generated_pixels(tmp_path):
    model = small_model().eval()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, quality_status="test-fixture")
    restored, metadata = load_checkpoint(path)
    assert metadata["quality_status"] == "test-fixture"
    assert torch.equal(sample_glyphs(model, "Ab0", 5), sample_glyphs(restored, "Ab0", 5))


def test_incompatible_label_contract_is_rejected(tmp_path):
    model = small_model()
    path = tmp_path / "bad.pt"
    save_checkpoint(path, model)
    payload = torch.load(path, weights_only=True)
    payload["alphabet"] = "ABC"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="alphabet"):
        load_checkpoint(path)


@pytest.mark.training
def test_a_fixed_batch_can_be_learned():
    torch.manual_seed(42)
    model = small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    images = torch.zeros(4, 1, 28, 28)
    images[:, :, 6:22, 12:16] = 1
    labels = torch.zeros(4, dtype=torch.long)
    latent = torch.zeros(4, 4)
    initial = torch.nn.functional.binary_cross_entropy_with_logits(
        model.decode(latent, labels), images
    )
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model.decode(latent, labels), images
        )
        loss.backward()
        optimizer.step()
    final = torch.nn.functional.binary_cross_entropy_with_logits(
        model.decode(latent, labels), images
    )
    assert final.item() < initial.item() * 0.6
