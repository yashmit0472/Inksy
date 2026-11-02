"""Small, from-scratch 28x28 cGAN; no encoder, pretrained weights or scorer loss."""

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm

from ml.characters import ALPHABET
from ml.model import LabelModulation


@dataclass(frozen=True)
class GANModelConfig:
    latent_dim: int = 32
    label_dim: int = 32
    base_channels: int = 32
    num_classes: int = len(ALPHABET)
    architecture: str = "cgan-projection-v1"

    def __post_init__(self):
        if min(self.latent_dim, self.label_dim, self.base_channels) < 1:
            raise ValueError("Model dimensions must be positive.")
        if self.num_classes != len(ALPHABET) or self.architecture != "cgan-projection-v1":
            raise ValueError("Unsupported cGAN architecture or alphabet.")

    def to_dict(self):
        return asdict(self)


class ConditionalGenerator(nn.Module):
    """Return foreground logits through the same decode interface as our VAE."""

    def __init__(self, config: GANModelConfig | None = None):
        super().__init__()
        self.config = config or GANModelConfig()
        c = self.config.base_channels
        self.label_embedding = nn.Embedding(self.config.num_classes, self.config.label_dim)
        self.input = nn.Linear(self.config.latent_dim + self.config.label_dim, c * 2 * 7 * 7)
        self.norm_7 = nn.GroupNorm(1, c * 2)
        self.condition_7 = LabelModulation(self.config.label_dim, c * 2)
        self.conv_14 = nn.Conv2d(c * 2, c, 3, padding=1)
        self.norm_14 = nn.GroupNorm(1, c)
        self.condition_14 = LabelModulation(self.config.label_dim, c)
        self.output = nn.Sequential(
            nn.Conv2d(c, max(4, c // 2), 3, padding=1),
            nn.GroupNorm(1, max(4, c // 2)),
            nn.SiLU(),
            nn.Conv2d(max(4, c // 2), 1, 3, padding=1),
        )

    def decode(self, latent: Tensor, labels: Tensor) -> Tensor:
        embedding = self.label_embedding(labels)
        features = self.input(torch.cat((latent, embedding), dim=1))
        features = features.reshape(-1, self.config.base_channels * 2, 7, 7)
        features = F.silu(self.condition_7(self.norm_7(features), embedding))
        features = self.conv_14(F.interpolate(features, scale_factor=2, mode="nearest"))
        features = F.silu(self.condition_14(self.norm_14(features), embedding))
        return self.output(F.interpolate(features, scale_factor=2, mode="nearest"))

    def forward(self, latent: Tensor, labels: Tensor) -> Tensor:
        return self.decode(latent, labels).sigmoid()


class ProjectionDiscriminator(nn.Module):
    """Spectrally normalized discriminator scoring image AND requested label."""

    def __init__(self, config: GANModelConfig | None = None):
        super().__init__()
        config = config or GANModelConfig()
        c = config.base_channels
        self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(1, c, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(c, c * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(c * 2, c * 4, 3, 2, 1)),
            nn.LeakyReLU(0.2),
        )
        self.unconditional = spectral_norm(nn.Linear(c * 4, 1))
        self.label_embedding = spectral_norm(nn.Embedding(config.num_classes, c * 4))

    def forward(self, images: Tensor, labels: Tensor) -> Tensor:
        # Both real EMNIST masks and generator probabilities arrive in [0, 1].
        features = self.features(images * 2 - 1).mean(dim=(2, 3))
        return self.unconditional(features).squeeze(1) + (
            features * self.label_embedding(labels)
        ).sum(1)


def discriminator_loss(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
    return F.relu(1 - real_scores).mean() + F.relu(1 + fake_scores).mean()


def generator_loss(fake_scores: Tensor) -> Tensor:
    return -fake_scores.mean()
