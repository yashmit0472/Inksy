"""From-scratch, single-writer text-to-word-image regression baseline."""

import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ml.iam import WORD_PREPROCESSING

FORMAT = "inksy-paired-word-generator-v1"
ROLE = "single-writer-paired-word-generator"
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def encode_texts(texts, device="cpu"):
    if not texts or len(texts) > 256:
        raise ValueError("Supply 1–256 words per batch.")
    if any(not isinstance(word, str) or not re.fullmatch(r"[a-z]{2,10}", word) for word in texts):
        raise ValueError("This experiment accepts only lowercase ASCII words of 2–10 letters.")
    lengths = torch.tensor([len(word) for word in texts], device=device)
    tokens = torch.zeros((len(texts), int(lengths.max())), dtype=torch.long, device=device)
    for index, word in enumerate(texts):
        tokens[index, : len(word)] = torch.tensor([ord(c) - 96 for c in word], device=device)
    return tokens, lengths


def column_mask(widths, height, canvas_width):
    valid = torch.arange(canvas_width, device=widths.device)[None, :] < widths[:, None]
    return valid[:, None, None, :].expand(-1, 1, height, -1)


def spatial_coordinates(widths, height, canvas_width, dtype):
    """Per-word coordinates at this resolution; batch padding never changes them."""
    x = (torch.arange(canvas_width, device=widths.device, dtype=dtype) + 0.5)[None, :]
    x = x / widths[:, None]
    y = (torch.arange(height, device=widths.device, dtype=dtype) + 0.5) / height
    x = x[:, None, :].expand(-1, height, -1)
    y = y[None, :, None].expand(len(widths), -1, canvas_width)
    planes = [2 * x - 1, 2 * y - 1]
    for frequency in (1, 2, 4, 8):
        for axis in (x, y):
            planes.extend(
                [
                    torch.sin(2 * math.pi * frequency * axis),
                    torch.cos(2 * math.pi * frequency * axis),
                ]
            )
    return torch.stack(planes, 1) * column_mask(widths, height, canvas_width)


@dataclass(frozen=True)
class WordGeneratorConfig:
    embedding: int = 32
    hidden: int = 64
    channels: int = 64
    architecture: str = "plain"
    writers: tuple[str, ...] = ()

    def __post_init__(self):
        for value in (self.embedding, self.hidden, self.channels):
            if type(value) is not int or not 8 <= value <= 128:
                raise ValueError("Model dimensions must be integers in [8, 128].")
        if self.architecture not in {"plain", "coordinate"}:
            raise ValueError("Unknown word-generator architecture.")
        if self.writers and (
            not isinstance(self.writers, tuple)
            or len(set(self.writers)) != len(self.writers)
            or any(not isinstance(writer, str) or not writer for writer in self.writers)
        ):
            raise ValueError("Writer IDs must be unique, nonempty strings in a tuple.")


class WordGenerator(nn.Module):
    """Text only at inference; target widths are optional training supervision.

    Packed contextual character features are interpolated across each word's
    horizontal grid. A global text feature and explicit coordinates accompany
    them. Convolutions predict vertical strokes jointly across the whole word.
    No image encoder, word-ID table, reference crop, OCR or pretrained component.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or WordGeneratorConfig()
        e, h, c = self.config.embedding, self.config.hidden, self.config.channels
        self.embedding = nn.Embedding(27, e, padding_idx=0)
        self.sequence = nn.GRU(e, h, batch_first=True, bidirectional=True)
        if self.config.writers:
            self.style = nn.Embedding(len(self.config.writers), 16)
            self.advance = nn.Linear(e + 16, 1)
            self.project = nn.Linear(2 * h + e + 16 + 5, c * 8)
        else:
            self.width_head = nn.Sequential(nn.Linear(2 * h + 1, h), nn.SiLU(), nn.Linear(h, 1))
            self.project = nn.Linear(4 * h + 3, c * 8)
        sizes = [c, c // 2, c // 4, max(8, c // 8)]
        extra = 18 if self.config.architecture == "coordinate" else 0
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Conv2d(a + extra, b, 3, padding=1), nn.Conv2d(b, b, 3, padding=1)]
                )
                for a, b in zip(sizes[:-1], sizes[1:], strict=True)
            ]
        )
        self.output = nn.Conv2d(sizes[-1], 1, 3, padding=1)
        nn.init.constant_(self.output.bias, -2.0)

    def forward(self, texts, widths=None, *, writers=None):
        tokens, lengths = encode_texts(texts, self.embedding.weight.device)
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed, _ = self.sequence(packed)
        sequence, _ = pad_packed_sequence(packed, batch_first=True)
        if self.config.writers:
            if writers is None or len(writers) != len(texts):
                raise ValueError("Supply one writer ID per word.")
            try:
                indices = [self.config.writers.index(writer) for writer in writers]
            except ValueError as error:
                raise ValueError("Unknown writer ID for this checkpoint.") from error
            style = self.style(torch.tensor(indices, device=tokens.device))
            conditioned = torch.cat([embedded, style[:, None].expand(-1, tokens.shape[1], -1)], -1)
            advances = (10 + 45 * self.advance(conditioned).squeeze(-1).sigmoid()) * (tokens != 0)
            predicted_widths = (8 + advances.sum(1)).clamp(max=384)
        else:
            if writers is not None:
                raise ValueError("This checkpoint uses a fixed writer.")
            global_text = sequence.sum(1) / lengths[:, None]
            width_input = torch.cat([global_text, lengths[:, None] / 10], dim=1)
            predicted_widths = 8 + 376 * self.width_head(width_input).squeeze(1).sigmoid()
        if widths is None:
            widths = (predicted_widths.detach() / 8).ceil().long().clamp(1, 48) * 8
        if (
            widths.dtype != torch.long
            or widths.shape != lengths.shape
            or widths.device != tokens.device
            or (widths < 8).any()
            or (widths > 384).any()
            or (widths % 8 != 0).any()
        ):
            raise ValueError("Widths must be one int64 multiple of eight in [8, 384] per word.")
        low_widths = widths // 8
        steps = torch.arange(int(low_widths.max()), device=tokens.device)
        x = (steps[None, :] + 0.5) / low_widths[:, None]
        if self.config.writers:
            # Monotone layout, with padding excluded from the learned character advances.
            bounds = advances.cumsum(1) / advances.sum(1, keepdim=True)
            slot = (x[:, :, None] >= bounds[:, None]).sum(-1).minimum((lengths - 1)[:, None])
            starts = F.pad(bounds[:, :-1], (1, 0))
            start, end = starts.gather(1, slot), bounds.gather(1, slot)
            phase = ((x - start) / (end - start).clamp_min(1e-5)).clamp(0, 1)
            position = (slot + phase - 0.5).clamp(min=0)
        else:
            position = (x * lengths[:, None] - 0.5).clamp(min=0)
        position = torch.minimum(position, (lengths - 1)[:, None])
        left = position.floor().long()
        right = torch.minimum(left + 1, (lengths - 1)[:, None])
        dimension = sequence.shape[-1]
        a = sequence.gather(1, left[:, :, None].expand(-1, -1, dimension))
        b = sequence.gather(1, right[:, :, None].expand(-1, -1, dimension))
        local = a + (b - a) * (position - left)[:, :, None]
        coordinates = torch.stack([x, torch.sin(math.pi * x), torch.cos(math.pi * x)], -1)
        if self.config.writers:
            char = embedded.gather(1, slot[:, :, None].expand(-1, -1, embedded.shape[-1]))
            within_char = torch.stack(
                [torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], -1
            )
            features = torch.cat(
                [
                    local,
                    char,
                    style[:, None].expand(-1, local.shape[1], -1),
                    coordinates,
                    within_char,
                ],
                -1,
            )
        else:
            features = torch.cat(
                [local, global_text[:, None].expand(-1, local.shape[1], -1), coordinates], -1
            )
        features = self.project(features).transpose(1, 2)
        features = features.reshape(len(texts), self.config.channels, 8, -1)
        features = features * column_mask(low_widths, 8, features.shape[-1])
        for block in self.blocks:
            features = F.interpolate(features, scale_factor=2, mode="nearest")
            low_widths = low_widths * 2
            mask = column_mask(low_widths, features.shape[-2], features.shape[-1])
            if self.config.architecture == "coordinate":
                coordinates = spatial_coordinates(
                    low_widths, features.shape[-2], features.shape[-1], features.dtype
                )
                features = torch.cat([features, coordinates], 1)
            for convolution in block:
                features = F.silu(convolution(features)) * mask
        logits = self.output(features)
        mask = column_mask(widths, 64, logits.shape[-1])
        logits = logits.masked_fill(~mask, -30)
        return {"logits": logits, "widths": widths, "predicted_widths": predicted_widths}


def adapt_generator(model, architecture):
    """Add zero-weight position inputs while preserving the trained function.

    This is structural migration of the project's own randomly initialized model,
    not training. Reverse migration discards learned inputs and is refused.
    """
    if architecture == model.config.architecture:
        return model
    if model.config.architecture != "plain" or architecture != "coordinate":
        raise ValueError("Only plain-to-coordinate migration is supported.")
    migrated = WordGenerator(replace(model.config, architecture=architecture))
    migrated.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    previous, state = model.state_dict(), migrated.state_dict()
    for name, value in state.items():
        source = previous[name]
        if value.shape == source.shape:
            value.copy_(source)
        elif name in {f"blocks.{index}.0.weight" for index in range(3)}:
            value.zero_()
            value[:, : source.shape[1]].copy_(source)
        else:
            raise ValueError(f"Unexpected migration shape: {name}")
    migrated.load_state_dict(state, strict=True)
    return migrated.train(model.training)


def paired_loss(result, targets, widths):
    logits = result["logits"]
    if logits.shape != targets.shape or logits.shape[1:3] != (1, 64):
        raise ValueError("Predictions and targets must share the B-by-1-by-64-by-W canvas.")
    if not torch.equal(widths, result["widths"]):
        raise ValueError("Paired loss requires target-aligned widths.")
    mask = column_mask(widths, 64, targets.shape[-1]).to(targets.dtype)
    pixels = logits.sigmoid()
    axes = (1, 2, 3)
    # Equal weight per word; long words do not dominate solely through pixel count.
    bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = ((bce_map * (1 + 3 * targets) * mask).sum(axes) / mask.sum(axes)).mean()
    intersection = (pixels * targets * mask).sum(axes)
    dice = (1 - (2 * intersection + 1) / (((pixels + targets) * mask).sum(axes) + 1)).mean()
    error = (pixels - targets).abs()
    foreground = (
        (error * targets * mask).sum(axes) / (targets.mul(mask).sum(axes).clamp_min(1))
    ).mean()
    background = (
        (error * (1 - targets) * mask).sum(axes) / (((1 - targets) * mask).sum(axes).clamp_min(1))
    ).mean()
    edge_terms = []
    for dim in (-1, -2):
        edge_mask = mask.narrow(dim, 1, mask.shape[dim] - 1) * mask.narrow(
            dim, 0, mask.shape[dim] - 1
        )
        delta = (pixels.diff(dim=dim) - targets.diff(dim=dim)).abs()
        edge_terms.append(((delta * edge_mask).sum(axes) / edge_mask.sum(axes).clamp_min(1)).mean())
    edge = sum(edge_terms) / 2
    width = F.smooth_l1_loss(result["predicted_widths"].log(), widths.float().log())
    total = bce + 0.5 * dice + edge + 0.2 * width
    return {
        "total": total,
        "bce": bce,
        "dice": dice,
        "edge": edge,
        "width": width,
        "foreground_mae": foreground,
        "background_mae": background,
    }


def save_generator(path, model, provenance):
    checkpoint = {
        "format": "inksy-multi-writer-v1" if model.config.writers else FORMAT,
        "role": "multi-writer-generator" if model.config.writers else ROLE,
        "alphabet": ALPHABET,
        "preprocessing": WORD_PREPROCESSING,
        "model_config": asdict(model.config),
        "state_dict": model.state_dict(),
        "provenance": provenance,
    }
    temporary = Path(path).with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_generator(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = WordGeneratorConfig(**checkpoint["model_config"])
    for key, expected in {
        "format": "inksy-multi-writer-v1" if config.writers else FORMAT,
        "role": "multi-writer-generator" if config.writers else ROLE,
        "alphabet": ALPHABET,
        "preprocessing": WORD_PREPROCESSING,
    }.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"Incompatible word-generator checkpoint: {key}")
    provenance = checkpoint.get("provenance", {})
    if not provenance.get("records_sha256") or (
        tuple(provenance.get("writers", ())) != config.writers
        if config.writers
        else not provenance.get("writer_id")
    ):
        raise ValueError("Checkpoint lacks writer/data provenance.")
    model = WordGenerator(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite weights.")
    return model.eval(), checkpoint
