"""Versioned, weights-only checkpoint loading with compatibility checks."""

from pathlib import Path

import torch

from ml.characters import ALPHABET, CONTRACT_VERSION, PREPROCESSING_VERSION
from ml.model import ConditionalVAE, ModelConfig

FORMAT_VERSION = 1


def load_checkpoint(path: Path, device: torch.device | str = "cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "format_version": FORMAT_VERSION,
        "alphabet": ALPHABET,
        "contract_version": CONTRACT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Incompatible checkpoint field: {key}")
    model = ConditionalVAE(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def save_checkpoint(path: Path, model: ConditionalVAE, **metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "alphabet": ALPHABET,
            "contract_version": CONTRACT_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "model_config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            **metadata,
        },
        temporary,
    )
    temporary.replace(path)
