"""Independent, from-scratch character recognizer used only for evaluation."""

from pathlib import Path

import torch
from torch import nn

from ml.characters import ALPHABET, CONTRACT_VERSION, PREPROCESSING_VERSION

RECOGNIZER_FORMAT = "inksy-recognizer-v1"


class CharacterRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            nn.GroupNorm(4, 24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.GroupNorm(8, 48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, len(ALPHABET)),
        )

    def forward(self, images):
        return self.network(images)


def confusion_metrics(confusion: torch.Tensor) -> dict:
    """Rows are requested/true labels, columns are predicted labels."""
    if confusion.shape != (len(ALPHABET), len(ALPHABET)):
        raise ValueError("Expected a 62-by-62 confusion matrix.")
    confusion = confusion.detach().cpu().long()
    counts = confusion.sum(1)
    present = counts > 0
    total = int(counts.sum())
    if not total:
        raise ValueError("Cannot score an empty confusion matrix.")
    correct = confusion.diag()
    casefold_correct = sum(
        int(confusion[i, j])
        for i, source in enumerate(ALPHABET)
        for j, observed in enumerate(ALPHABET)
        if source.casefold() == observed.casefold()
    )
    return {
        "examples": total,
        "accuracy": int(correct.sum()) / total,
        "macro_accuracy": float((correct[present] / counts[present]).double().mean()),
        "case_insensitive_accuracy": casefold_correct / total,
        "classes_evaluated": int(present.sum()),
        "per_class": {
            char: {
                "count": int(counts[i]),
                "accuracy": int(correct[i]) / int(counts[i]) if counts[i] else None,
            }
            for i, char in enumerate(ALPHABET)
        },
        "confusion_matrix": confusion.tolist(),
    }


def load_recognizer(path: Path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "format": RECOGNIZER_FORMAT,
        "alphabet": ALPHABET,
        "contract_version": CONTRACT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Incompatible recognizer checkpoint: {key}")
    model = CharacterRecognizer()
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload
