"""Create the printable worksheet and extract cells from an aligned photo."""

import base64
import binascii
import io
from functools import cache

import numpy as np
from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageFilter,
    ImageFont,
    UnidentifiedImageError,
)

from rendering.personal_styles import FULL_OWNER_LABELS, SHEET_SIZE

PAGE_SIZE = (1240, 1754)
MARKERS = ((42, 42), (1198, 42), (1198, 1712), (42, 1712))
CELLS = tuple((90 + i % 8 * 137, 155 + i // 8 * 95) for i in range(len(FULL_OWNER_LABELS)))
BOX = 74


@cache
def template():
    page = Image.new("L", PAGE_SIZE, 255)
    draw = ImageDraw.Draw(page)
    draw.text((83, 53), "Inksy handwriting worksheet", fill=0, font=ImageFont.load_default(30))
    draw.text(
        (83, 96),
        "Use a dark pen. Write one character inside each box. Keep all four markers visible.",
        fill=0,
        font=ImageFont.load_default(18),
    )
    for x, y in MARKERS:
        draw.rectangle((x - 15, y - 15, x + 15, y + 15), fill=0)
    label = ImageFont.load_default(19)
    for i, ((x, y), char) in enumerate(zip(CELLS, FULL_OWNER_LABELS, strict=True)):
        draw.text((x, y - 23), f"{char}  ({i % 2 + 1})", fill=0, font=label)
        draw.rectangle((x, y, x + BOX, y + BOX), outline=100, width=2)
    stream = io.BytesIO()
    page.save(stream, format="PNG")
    return stream.getvalue()


def perspective(points, size):
    if (
        not isinstance(points, list)
        or len(points) != 4
        or any(
            not isinstance(point, list)
            or len(point) != 2
            or any(type(value) not in (int, float) or not 0 <= value <= 1 for value in point)
            for point in points
        )
    ):
        raise ValueError("Select all four markers in the order shown.")
    scaled = [(x * size[0], y * size[1]) for x, y in points]
    for i, (ax, ay) in enumerate(scaled):
        bx, by = scaled[(i + 1) % 4]
        cx, cy = scaled[(i + 2) % 4]
        if (bx - ax) * (cy - by) - (by - ay) * (cx - bx) < size[0] * size[1] * 0.04:
            raise ValueError("Select the marker centers clockwise, starting at the top left.")
    equations, values = [], []
    for (x, y), (u, v) in zip(MARKERS, scaled, strict=True):
        equations.extend(((x, y, 1, 0, 0, 0, -u * x, -u * y), (0, 0, 0, x, y, 1, -v * x, -v * y)))
        values.extend((u, v))
    try:
        return tuple(np.linalg.solve(equations, values))
    except np.linalg.LinAlgError as error:
        raise ValueError("The markers could not be aligned. Select them again.") from error


def crop_sample(page, x, y):
    # Exclude the printed border and estimate the paper tone around pen strokes.
    gray = page.crop((x + 5, y + 5, x + BOX - 4, y + BOX - 4))
    gray = gray.resize((88, 88), Image.Resampling.BICUBIC)
    background = gray.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(5))
    ink = ImageChops.subtract(background, gray)
    peak = ink.getextrema()[1]
    if peak < 55:
        return None
    mask = ink.point(lambda value: 255 if value >= max(40, peak // 3) else 0)
    bounds = mask.getbbox()
    if bounds is None or bounds[3] - bounds[1] < 8 or not 18 <= mask.histogram()[255] <= 4000:
        return None
    if min(bounds[:2]) < 1 or max(bounds[2:]) > 87:
        return None
    cleaned = ink.point(lambda value: 255 - round(255 * max(0, value - 35) / (peak - 35)))
    sample = Image.new("L", SHEET_SIZE, 255)
    sample.paste(cleaned, (4, 4))
    stream = io.BytesIO()
    sample.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def photo_samples(encoded, points):
    if not isinstance(encoded, str) or len(encoded) > 5_500_000:
        raise ValueError("Choose a worksheet photo under 4 MB after resizing.")
    try:
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as photo:
            if (
                photo.format not in {"JPEG", "PNG"}
                or max(photo.size) > 2600
                or min(photo.size) < 500
            ):
                raise ValueError("Use a clear photo of the full worksheet at 500 pixels or larger.")
            image = photo.convert("L")
    except (binascii.Error, OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise ValueError("The photo could not be opened. Try a JPEG or PNG.") from error
    page = image.transform(
        PAGE_SIZE,
        Image.Transform.PERSPECTIVE,
        perspective(points, image.size),
        resample=Image.Resampling.BICUBIC,
        fillcolor=255,
    )
    for x, y in MARKERS:
        marker = page.crop((x - 5, y - 5, x + 6, y + 6))
        if sum(marker.histogram()[:100]) < 60:
            raise ValueError("A corner marker was missed. Select the center of each black square.")
    samples = [crop_sample(page, x, y) for x, y in CELLS]
    if not any(samples):
        raise ValueError("No handwriting was found. Check alignment and use a dark pen.")
    return samples
