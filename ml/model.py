"""Class-conditional convolutional VAE, initialized from scratch."""

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ml.characters import ALPHABET


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 32
    label_dim: int = 16
    base_channels: int = 32
    num_classes: int = len(ALPHABET)
    architecture: str = "baseline"

    def __post_init__(self):
        if min(self.latent_dim, self.label_dim, self.base_channels) < 1:
            raise ValueError("Model dimensions must be positive.")
        if self.num_classes != len(ALPHABET):
            raise ValueError("This checkpoint contract requires all 62 ByClass labels.")
        if self.architecture not in {"baseline", "refined"}:
            raise ValueError("Architecture must be baseline or refined.")

    def to_dict(self) -> dict:
        return asdict(self)


class LabelModulation(nn.Module):
    """Feature-wise affine conditioning, identity-initialized for safe warm starts."""

    def __init__(self, label_dim: int, channels: int):
        super().__init__()
        self.affine = nn.Linear(label_dim, 2 * channels)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, features: Tensor, embedding: Tensor) -> Tensor:
        scale, shift = self.affine(embedding).chunk(2, dim=1)
        return features * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class ConditionalVAE(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config.base_channels
        self.label_embedding = nn.Embedding(self.config.num_classes, self.config.label_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.SiLU(),
            nn.Flatten(),
        )
        features = c * 2 * 7 * 7
        self.encoder_head = nn.Sequential(
            nn.Linear(features + self.config.label_dim, 256), nn.SiLU()
        )
        self.mu = nn.Linear(256, self.config.latent_dim)
        self.logvar = nn.Linear(256, self.config.latent_dim)
        self.decoder_input = nn.Sequential(
            nn.Linear(self.config.latent_dim + self.config.label_dim, features), nn.SiLU()
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (c * 2, 7, 7)),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(c, 1, 4, 2, 1),
        )
        if self.config.architecture == "refined":
            self.condition_7 = LabelModulation(self.config.label_dim, c * 2)
            self.condition_14 = LabelModulation(self.config.label_dim, c)
            self.stroke_refiner = nn.Sequential(
                nn.Conv2d(c, max(4, c // 2), 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(max(4, c // 2), 1, 3, padding=1),
            )
            # A transferred baseline produces exactly the same logits initially.
            nn.init.zeros_(self.stroke_refiner[-1].weight)
            nn.init.zeros_(self.stroke_refiner[-1].bias)

    def encode(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encoder(images)
        hidden = self.encoder_head(torch.cat((features, self.label_embedding(labels)), dim=1))
        return self.mu(hidden), self.logvar(hidden).clamp(-20, 20)

    @staticmethod
    def reparameterize(
        mu: Tensor, logvar: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        noise = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
        return mu + (0.5 * logvar).exp() * noise

    def decode(self, latent: Tensor, labels: Tensor) -> Tensor:
        """Return logits; callers apply sigmoid for foreground intensities."""
        embedding = self.label_embedding(labels)
        hidden = self.decoder_input(torch.cat((latent, embedding), dim=1))
        if self.config.architecture == "baseline":
            return self.decoder(hidden)
        features_7 = self.condition_7(self.decoder[0](hidden), embedding)
        features_14 = self.condition_14(self.decoder[2](self.decoder[1](features_7)), embedding)
        logits = self.decoder[3](features_14)
        refined_features = F.interpolate(features_14, scale_factor=2, mode="nearest")
        return logits + self.stroke_refiner(refined_features)

    def forward(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(images, labels)
        return self.decode(self.reparameterize(mu, logvar), labels), mu, logvar


def vae_loss(
    logits: Tensor,
    targets: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
    edge_weight: float = 0.0,
):
    """Pixel/latent sums per example; optional signed image-gradient preservation term."""
    if beta < 0 or edge_weight < 0:
        raise ValueError("Loss weights cannot be negative.")
    reconstruction = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    reconstruction = reconstruction.flatten(1).sum(1).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    probabilities = logits.sigmoid()
    horizontal = (torch.diff(probabilities, dim=3) - torch.diff(targets, dim=3)).abs()
    vertical = (torch.diff(probabilities, dim=2) - torch.diff(targets, dim=2)).abs()
    edge = 0.5 * (horizontal.flatten(1).sum(1) + vertical.flatten(1).sum(1)).mean()
    base_loss = reconstruction + beta * kl
    return {
        "loss": base_loss + edge_weight * edge,
        "base_loss": base_loss,
        "reconstruction": reconstruction,
        "kl": kl,
        "edge": edge,
    }
