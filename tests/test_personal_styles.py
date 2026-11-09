import base64
import io

import pytest
from PIL import Image, ImageDraw

from rendering.personal_styles import FULL_OWNER_LABELS, worksheet_targets
from rendering.worksheet_photo import CELLS, MARKERS, PAGE_SIZE, photo_samples, template


def worksheet_png(*, ink):
    image = Image.new("RGBA", (96, 96))
    if ink:
        ImageDraw.Draw(image).line((24, 68, 40, 24, 56, 68), fill="black", width=3)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def test_transparent_worksheet_is_read_as_ink_and_blank_box_is_identified():
    samples = [worksheet_png(ink=True)] * 52
    targets, widths, hashes = worksheet_targets(samples)
    assert targets.shape == (52, 1, 48, 48)
    assert targets.max() > 0.9
    assert len(widths) == len(hashes) == 52
    samples[1] = worksheet_png(ink=False)
    with pytest.raises(ValueError, match="a, sample 2"):
        worksheet_targets(samples)


def test_photo_alignment_preserves_digit_slot_and_leaves_empty_boxes_unfilled():
    page = Image.open(io.BytesIO(template())).convert("L")
    x, y = CELLS[-1]
    ImageDraw.Draw(page).line((x + 24, y + 22, x + 51, y + 22, x + 33, y + 56), fill=0, width=4)
    photo = Image.new("L", (PAGE_SIZE[0] + 100, PAGE_SIZE[1] + 100), 220)
    photo.paste(page, (30, 55))
    points = [[(x + 30) / photo.width, (y + 55) / photo.height] for x, y in MARKERS]
    stream = io.BytesIO()
    photo.save(stream, format="JPEG")
    payload = base64.b64encode(stream.getvalue()).decode("ascii")
    samples = photo_samples(payload, points)
    assert FULL_OWNER_LABELS[-1] == "9"
    assert len(samples) == 124 and all(sample is None for sample in samples[:-1])
    targets, widths, _ = worksheet_targets([samples[-1]], ("9",))
    assert targets.max() > 0.9 and len(widths) == 1
    with pytest.raises(ValueError, match="clockwise"):
        photo_samples(payload, points[::-1])
