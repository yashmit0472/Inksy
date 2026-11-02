"""Versioned, weights-only checkpoint loading with compatibility checks."""

from pathlib import Path

import torch

from ml.characters import ALPHABET, CONTRACT_VERSION, PREPROCESSING_VERSION
from ml.gan import ConditionalGenerator, GANModelConfig
from ml.model import ConditionalVAE, ModelConfig
from ml.utils import sha256_file

FORMAT_VERSION = 2


def load_checkpoint(path: Path, device: torch.device | str = "cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "alphabet": ALPHABET,
        "contract_version": CONTRACT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    if checkpoint.get("format_version") not in {1, FORMAT_VERSION}:
        raise ValueError("Incompatible checkpoint field: format_version")
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Incompatible checkpoint field: {key}")
    family = checkpoint.get("model_family", "vae")
    if family == "vae":
        model = ConditionalVAE(ModelConfig(**checkpoint["model_config"]))
    elif family == "cgan":
        model = ConditionalGenerator(GANModelConfig(**checkpoint["model_config"]))
    else:
        raise ValueError(f"Unsupported checkpoint model_family: {family}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def initialize_from_checkpoint(model: ConditionalVAE, path: Path, expected_split: str) -> dict:
    """Copy our own learned weights; do not restore optimizer state or mutate the source file."""
    source, metadata = load_checkpoint(path)
    if not isinstance(source, ConditionalVAE):
        raise ValueError("VAE initialization requires a VAE checkpoint, not a cGAN.")
    if metadata.get("split_sha256") != expected_split:
        raise ValueError("Initialization checkpoint uses a different data split.")
    dimensions = ("latent_dim", "label_dim", "base_channels", "num_classes")
    if any(getattr(source.config, key) != getattr(model.config, key) for key in dimensions):
        raise ValueError("Initialization checkpoint has incompatible model dimensions.")
    if source.config.architecture == "refined" and model.config.architecture == "baseline":
        raise ValueError("Cannot initialize a baseline by discarding learned refinement layers.")
    result = model.load_state_dict(source.state_dict(), strict=False)
    allowed_prefixes = ("condition_7.", "condition_14.", "stroke_refiner.")
    if result.unexpected_keys or any(
        not key.startswith(allowed_prefixes) for key in result.missing_keys
    ):
        raise ValueError("Unexpected initialization checkpoint parameter mismatch.")
    return {
        "checkpoint": str(path),
        "sha256": sha256_file(path),
        "source_epoch": metadata.get("epoch"),
        "source_architecture": source.config.architecture,
        "new_parameters": list(result.missing_keys),
        "optimizer_restored": False,
        "note": "Weight initialization from our checkpoint; this is not interrupted-run resume.",
    }


def save_checkpoint(path: Path, model: ConditionalVAE | ConditionalGenerator, **metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "model_family": "cgan" if isinstance(model, ConditionalGenerator) else "vae",
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
