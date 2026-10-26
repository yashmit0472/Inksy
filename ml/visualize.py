"""Pillow sample grids. Images use white paper and dark ink for visual inspection."""

import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def save_grid(images: torch.Tensor, path: Path, labels: list[str] | None = None, columns=12):
    images = images.detach().cpu().clamp(0, 1)
    if len(images) == 0:
        raise ValueError("Cannot render an empty grid.")
    cell_w, cell_h = 64, 78
    columns = min(columns, len(images))
    grid = Image.new(
        "RGB", (columns * cell_w, math.ceil(len(images) / columns) * cell_h), "#f1f4f8"
    )
    draw = ImageDraw.Draw(grid)
    for index, tensor in enumerate(images):
        ink = (255 * (1 - tensor.squeeze())).round().to(torch.uint8).numpy()
        image = Image.fromarray(ink).resize((48, 48), Image.Resampling.NEAREST).convert("RGB")
        x, y = (index % columns) * cell_w, (index // columns) * cell_h
        grid.paste(image, (x + 8, y + 5))
        if labels:
            draw.text((x + 8, y + 57), labels[index], fill="#17283e")
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
