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

    def __post_init__(self):
        if min(self.latent_dim, self.label_dim, self.base_channels) < 1:
            raise ValueError("Model dimensions must be positive.")
        if self.num_classes != len(ALPHABET):
            raise ValueError("This checkpoint contract requires all 62 ByClass labels.")

    def to_dict(self) -> dict:
        return asdict(self)


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
        hidden = self.decoder_input(torch.cat((latent, self.label_embedding(labels)), dim=1))
        return self.decoder(hidden)

    def forward(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(images, labels)
        return self.decode(self.reparameterize(mu, logvar), labels), mu, logvar


def vae_loss(logits: Tensor, targets: Tensor, mu: Tensor, logvar: Tensor, beta: float = 1.0):
    """Sum over pixels/latent dimensions, mean over examples for both loss terms."""
    reconstruction = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    reconstruction = reconstruction.flatten(1).sum(1).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    return {"loss": reconstruction + beta * kl, "reconstruction": reconstruction, "kl": kl}
