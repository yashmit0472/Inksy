"""From-scratch CTC word reader for the future generator's content loss."""

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ml.iam import WORD_ALPHABET, WORD_CONTRACT, WORD_PREPROCESSING, ctc_min_steps
from ml.utils import resolve_device, sha256_file, write_json

READER_FORMAT = "inksy-word-reader-v1"
READER_ROLE = "training-content-reader"


@dataclass(frozen=True)
class WordReaderConfig:
    channels: int = 24
    hidden: int = 128
    layers: int = 2
    architecture: str = "mean"

    def __post_init__(self):
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (self.channels, self.hidden, self.layers)
        ):
            raise ValueError("Word-reader dimensions must be positive integers.")
        if self.channels > 128 or self.hidden > 512 or self.layers > 4:
            raise ValueError("Word-reader dimensions exceed the bounded architecture contract.")
        if self.architecture not in {"mean", "spatial"}:
            raise ValueError("Unknown word-reader architecture.")


def mask_columns(features, widths):
    columns = torch.arange(features.shape[-1], device=features.device)
    mask = columns[None, :] < widths.to(features.device)[:, None]
    return features * mask[:, None, None, :]


class WordReader(nn.Module):
    def __init__(self, config: WordReaderConfig | None = None):
        super().__init__()
        self.config = config or WordReaderConfig()
        c = self.config.channels
        self.convolutions = nn.ModuleList(
            [
                nn.Conv2d(1, c, 3, padding=1),
                nn.Conv2d(c, c * 2, 3, padding=1),
                nn.Conv2d(c * 2, c * 4, 3, padding=1),
            ]
        )
        if self.config.architecture == "spatial":
            # Preserve all existing predictions at initialization. Learning can then
            # distinguish vertical positions instead of uniformly averaging them.
            self.height_projection = nn.Conv2d(c * 4, c * 4, (16, 1), bias=False)
            nn.init.zeros_(self.height_projection.weight)
        self.sequence = nn.GRU(
            c * 4,
            self.config.hidden,
            num_layers=self.config.layers,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(self.config.hidden * 2, len(WORD_ALPHABET) + 1)

    def forward(self, images, widths):
        if images.ndim != 4 or images.shape[1:3] != (1, 64) or len(images) == 0:
            raise ValueError("Expected nonempty B-by-1-by-64-by-W foreground images.")
        if widths.ndim != 1 or len(widths) != len(images) or widths.dtype != torch.long:
            raise ValueError("Each word needs one int64 width.")
        if (widths < 8).any() or (widths > images.shape[-1]).any() or (widths % 4).any():
            raise ValueError("Word widths must fit the canvas and be divisible by four.")
        features = mask_columns(images, widths)
        current_widths = widths
        for index, convolution in enumerate(self.convolutions):
            features = mask_columns(F.silu(convolution(features)), current_widths)
            if index < 2:
                features = F.max_pool2d(features, 2)
                current_widths = current_widths // 2
                features = mask_columns(features, current_widths)
        # Column masks stop convolution biases leaking through the right-hand batch padding.
        features = self.column_features(features).transpose(1, 2)
        packed = pack_padded_sequence(
            features, current_widths.cpu(), batch_first=True, enforce_sorted=False
        )
        sequence, _ = self.sequence(packed)
        sequence, _ = pad_packed_sequence(
            sequence, batch_first=True, total_length=features.shape[1]
        )
        return self.classifier(sequence).log_softmax(-1).transpose(0, 1), current_widths.cpu()

    def column_features(self, features):
        columns = features.mean(dim=2)
        if self.config.architecture == "spatial":
            columns = columns + self.height_projection(features).squeeze(2) / features.shape[2]
        return columns


def adapt_reader(model, target_config):
    """Explicit own-weight migration; old mean checkpoints remain strictly loadable."""
    if model.config == target_config:
        return model
    source_dimensions = (model.config.channels, model.config.hidden, model.config.layers)
    target_dimensions = (target_config.channels, target_config.hidden, target_config.layers)
    if (
        source_dimensions != target_dimensions
        or model.config.architecture != "mean"
        or target_config.architecture != "spatial"
    ):
        raise ValueError("Only same-dimension mean-to-spatial reader migration is supported.")
    upgraded = WordReader(target_config).to(next(model.parameters()).device)
    missing, unexpected = upgraded.load_state_dict(model.state_dict(), strict=False)
    if missing != ["height_projection.weight"] or unexpected:
        raise ValueError("Unexpected parameters during reader migration.")
    return upgraded.train(model.training)


def word_ctc_loss(log_probabilities, input_lengths, batch, *, sample_weights=None):
    targets = batch["ctc_targets"]
    target_lengths = batch["token_lengths"].cpu()
    required = torch.tensor([ctc_min_steps(text) for text in batch["text"]])
    if (input_lengths.cpu() < required).any():
        raise ValueError("CTC time axis cannot align the repeated-character targets.")
    if (targets <= 0).any() or (targets > len(WORD_ALPHABET)).any():
        raise ValueError("CTC targets must not contain padding/blank or unknown tokens.")
    if int(target_lengths.sum()) != len(targets):
        raise ValueError("CTC target lengths do not match the flattened targets.")
    # Native CPU CTC keeps strict determinism and remains differentiable across device copies.
    # CUDA CTC can be nondeterministic; the project deliberately keeps that path off.
    if sample_weights is not None:
        if (
            sample_weights.ndim != 1
            or len(sample_weights) != len(target_lengths)
            or not torch.isfinite(sample_weights).all()
            or (sample_weights <= 0).any()
        ):
            raise ValueError("CTC sample weights must be finite positive values for every word.")
    losses = F.ctc_loss(
        log_probabilities.float().cpu(),
        targets.cpu(),
        input_lengths.cpu(),
        target_lengths,
        blank=0,
        reduction="mean" if sample_weights is None else "none",
        zero_infinity=False,
    )
    if sample_weights is None:
        return losses
    # Preserve the usual per-target-length CTC normalization, then reweight words.
    weights = sample_weights.detach().to(device="cpu", dtype=losses.dtype)
    return ((losses / target_lengths) * weights).sum() / weights.sum()


def greedy_decode(log_probabilities, lengths):
    if log_probabilities.ndim != 3 or log_probabilities.shape[2] != len(WORD_ALPHABET) + 1:
        raise ValueError("Unexpected CTC alphabet or log-probability shape.")
    if (
        len(lengths) != log_probabilities.shape[1]
        or (lengths < 1).any()
        or (lengths > log_probabilities.shape[0]).any()
    ):
        raise ValueError("Invalid CTC decoding lengths.")
    paths = log_probabilities.detach().argmax(2).cpu().transpose(0, 1)
    result = []
    for path, length in zip(paths, lengths, strict=True):
        previous, text = 0, []
        for token in path[: int(length)].tolist():
            if token and token != previous:
                text.append(WORD_ALPHABET[token - 1])
            previous = token
        result.append("".join(text))
    return result


def edit_distance(expected: str, observed: str) -> int:
    previous = list(range(len(observed) + 1))
    for i, source in enumerate(expected, 1):
        current = [i]
        for j, target in enumerate(observed, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (source != target))
            )
        previous = current
    return previous[-1]


def reading_metrics(rows):
    if not rows or any(not row["expected"] for row in rows):
        raise ValueError("Reading metrics require nonempty reference words.")
    characters = sum(len(row["expected"]) for row in rows)
    errors = sum(edit_distance(row["expected"], row["predicted"]) for row in rows)
    correct = sum(row["expected"] == row["predicted"] for row in rows)
    return {
        "examples": len(rows),
        "reference_characters": characters,
        "character_edits": errors,
        "character_error_rate": errors / characters,
        "exact_word_accuracy": correct / len(rows),
        "word_error_rate": 1 - correct / len(rows),
        "case_insensitive_word_accuracy": sum(
            row["expected"].casefold() == row["predicted"].casefold() for row in rows
        )
        / len(rows),
        "empty_prediction_fraction": sum(not row["predicted"] for row in rows) / len(rows),
    }


def save_reader(path, model, **metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format": READER_FORMAT,
            "role": READER_ROLE,
            "word_contract": WORD_CONTRACT,
            "preprocessing": WORD_PREPROCESSING,
            "alphabet": WORD_ALPHABET,
            "model_config": asdict(model.config),
            "state_dict": model.state_dict(),
            **metadata,
        },
        temporary,
    )
    temporary.replace(path)


def load_reader(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key, expected in {
        "format": READER_FORMAT,
        "role": READER_ROLE,
        "word_contract": WORD_CONTRACT,
        "preprocessing": WORD_PREPROCESSING,
        "alphabet": WORD_ALPHABET,
    }.items():
        if payload.get(key) != expected:
            raise ValueError(f"Incompatible word-reader checkpoint: {key}")
    model = WordReader(WordReaderConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


def write_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "expected", "predicted"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Evaluate real development words only; never launches optimization."""
    from ml.train_word_reader import evaluation_loader, reader_epoch

    parser = argparse.ArgumentParser(description="Inference-only word-reader evaluation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/iam-words"))
    parser.add_argument(
        "--partition",
        choices=["seen_validation", "content_validation"],
        default="content_validation",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new word-reader evaluation output directory.")
    torch.set_num_threads(4)
    device = resolve_device(args.device)
    model, metadata = load_reader(args.checkpoint, device)
    loader, identity = evaluation_loader(args.prepared, args.partition, 16, args.max_batches)
    if metadata["records_sha256"] != identity["records_sha256"]:
        raise ValueError("Reader and evaluation data use different prepared records.")
    metrics, rows = reader_epoch(model, loader, device, description=args.partition)
    args.output.mkdir(parents=True, exist_ok=False)
    write_predictions(args.output / "predictions.csv", rows)
    write_json(
        args.output / "evaluation.json",
        {
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "partition": identity,
            "metrics": metrics,
            "role": READER_ROLE,
            "training_performed": False,
            "test_set_evaluated": False,
            "warning": (
                "Development diagnostics of a training-loss reader, "
                "not independent generator evaluation."
            ),
        },
    )
    print(
        f"CER={metrics['character_error_rate']:.2%}; "
        f"exact words={metrics['exact_word_accuracy']:.2%}"
    )


if __name__ == "__main__":
    main()
