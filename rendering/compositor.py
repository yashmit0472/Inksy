"""Proportional isolated-glyph page layout. No font fallback for model characters."""

import math
import re
from dataclasses import dataclass
from functools import lru_cache

from PIL import Image, ImageDraw

from ml.characters import CHAR_TO_LABEL, PUNCTUATION, validate_text

RENDERER_VERSION = "isolated-glyph-pages-v1"
MAX_PAGES = 3
MAX_TOTAL_PIXELS = 7_000_000


@dataclass(frozen=True)
class LayoutOptions:
    width: int = 1240
    height: int = 1754
    margin: int = 80
    glyph_size: int = 32
    character_spacing: int = 2
    word_spacing: int = 0
    line_spacing: int = 12
    paper: str = "ruled"

    def __post_init__(self):
        integer_fields = (
            self.width,
            self.height,
            self.margin,
            self.glyph_size,
            self.character_spacing,
            self.word_spacing,
            self.line_spacing,
        )
        if any(type(value) is not int for value in integer_fields):
            raise ValueError("Layout dimensions and spacing must be integers.")
        if not 320 <= self.width <= 2048 or not 320 <= self.height <= 2048:
            raise ValueError("Page dimensions must be between 320 and 2048 pixels.")
        if not 16 <= self.margin < min(self.width, self.height) // 2:
            raise ValueError("Margins leave no valid writing area.")
        if not 12 <= self.glyph_size <= 64:
            raise ValueError("Glyph size must be between 12 and 64 pixels.")
        if not 0 <= self.character_spacing <= 24 or not 0 <= self.word_spacing <= 48:
            raise ValueError("Character spacing must be 0-24 and word spacing 0-48 pixels.")
        if not 0 <= self.line_spacing <= 64:
            raise ValueError("Line spacing must be 0-64 pixels.")
        if self.paper not in {"plain", "ruled", "grid"}:
            raise ValueError("Paper must be plain, ruled or grid.")
        if self.height - 2 * self.margin < self.glyph_size + self.descent:
            raise ValueError("The writing area cannot fit even one line.")

    @property
    def descent(self):
        return math.ceil(0.28 * self.glyph_size)

    @property
    def line_step(self):
        return self.glyph_size + self.descent + self.line_spacing

    @property
    def space_advance(self):
        return max(4, round(self.glyph_size * 0.42)) + self.word_spacing


@dataclass(frozen=True)
class Glyph:
    mask: Image.Image
    advance: int
    baseline_offset: int
    kind: str


@dataclass(frozen=True)
class Placement:
    index: int
    character: str
    page: int
    line: int
    x: int
    y: int
    baseline: int
    width: int
    height: int
    kind: str


@dataclass
class PageLayout:
    placements: list[Placement]
    glyphs: dict[int, Glyph]
    page_count: int


def model_glyph(character: str, mask: Image.Image, options: LayoutOptions) -> Glyph:
    if character not in CHAR_TO_LABEL:
        raise ValueError("Only alphanumerics can use a model-generated glyph.")
    if mask.mode != "L" or mask.size != (28, 28):
        raise ValueError("Expected a 28x28 grayscale foreground mask from the model.")
    # Threshold only to find bounds; keep the original gray values inside the padded crop.
    bounds = mask.point(lambda pixel: 255 if pixel > 25 else 0).getbbox()
    if bounds:
        left, top, right, bottom = bounds
        mask = mask.crop(
            (max(0, left - 1), max(0, top - 1), min(28, right + 1), min(28, bottom + 1))
        )
    size = options.glyph_size
    if character.isupper() or character.isdigit():
        height, below = size, 0
    elif character in "gjpqy":
        height, below = round(size * (1.08 if character == "j" else 0.96)), round(size * 0.24)
    elif character in "bdfhklt":
        height, below = round(size * 0.96), 0
    elif character == "i":
        height, below = round(size * 0.8), 0
    else:
        height, below = round(size * 0.72), 0
    scale = min(height / mask.height, (size * 1.4) / mask.width)
    width = max(1, round(mask.width * scale))
    height = max(1, round(mask.height * scale))
    resized = mask.resize((width, height), Image.Resampling.LANCZOS)
    advance = max(width, round(size * 0.18)) + options.character_spacing
    return Glyph(resized, advance, below - height, "model")


@lru_cache(maxsize=256)
def punctuation_glyph(character: str, size: int, spacing: int) -> Glyph:
    if character not in PUNCTUATION:
        raise ValueError("Unsupported procedural punctuation.")
    scale = 4
    canvas = Image.new("L", (math.ceil(size * 0.6 * scale), math.ceil(size * 1.3 * scale)), 0)
    draw = ImageDraw.Draw(canvas)
    s = size * scale
    center = 0.29 * s
    thickness = max(1, round(0.065 * s))
    radius = max(1, 0.055 * s)

    def dot(x, y):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

    if character in ".,;:":
        dot(center, 0.94 * s)
        if character in ":;":
            dot(center, 0.48 * s)
        if character in ",;":
            draw.line(
                [(center + radius * 0.5, 0.97 * s), (center - radius, 1.12 * s)],
                fill=255,
                width=thickness,
            )
    elif character == "-":
        draw.line([(0.06 * s, 0.64 * s), (0.51 * s, 0.64 * s)], fill=255, width=thickness)
    elif character in "'\"":
        positions = [center] if character == "'" else [0.17 * s, 0.4 * s]
        for x in positions:
            draw.line([(x, 0.08 * s), (x - 0.04 * s, 0.28 * s)], fill=255, width=thickness)
    elif character == "!":
        draw.line([(center, 0.07 * s), (center, 0.71 * s)], fill=255, width=thickness)
        dot(center, 0.94 * s)
    elif character == "?":
        draw.arc((0.06 * s, 0.06 * s, 0.52 * s, 0.55 * s), 180, 450, fill=255, width=thickness)
        draw.line([(center, 0.55 * s), (center, 0.73 * s)], fill=255, width=thickness)
        dot(center, 0.94 * s)
    elif character == "(":
        draw.arc((0.12 * s, 0.03 * s, 0.64 * s, 1.13 * s), 90, 270, fill=255, width=thickness)
    else:
        draw.arc((-0.1 * s, 0.03 * s, 0.42 * s, 1.13 * s), 270, 450, fill=255, width=thickness)
    bounds = canvas.getbbox()
    if bounds is None:
        raise ValueError("Procedural punctuation produced an empty mask.")
    left, top, right, bottom = bounds
    cropped = canvas.crop(bounds)
    width, height = (
        max(1, math.ceil(cropped.width / scale)),
        max(1, math.ceil(cropped.height / scale)),
    )
    mask = cropped.resize((width, height), Image.Resampling.LANCZOS)
    return Glyph(mask, width + spacing + 1, round(top / scale) - size, "procedural")


def layout_text(text: str, masks: dict[int, Image.Image], options: LayoutOptions) -> PageLayout:
    validate_text(text)
    glyphs = {}
    for index, character in enumerate(text):
        if character in CHAR_TO_LABEL:
            if index not in masks:
                raise ValueError(f"Missing generated glyph at input position {index}.")
            glyphs[index] = model_glyph(character, masks[index], options)
        elif character in PUNCTUATION:
            glyphs[index] = punctuation_glyph(
                character, options.glyph_size, options.character_spacing
            )
    placements = []
    left, right = options.margin, options.width - options.margin
    x, baseline, page, line = left, options.margin + options.glyph_size, 0, 0

    def next_line():
        nonlocal x, baseline, page, line
        x = left
        baseline += options.line_step
        line += 1
        if baseline + options.descent > options.height - options.margin:
            page += 1
            baseline = options.margin + options.glyph_size
            line = 0

    for token in re.finditer(r"\n| +|[^ \n]+", text):
        content = token.group()
        if content == "\n":
            next_line()
            continue
        if content.isspace():
            for _ in content:
                if x + options.space_advance > right:
                    next_line()
                x += options.space_advance
            continue
        indices = list(range(token.start(), token.end()))
        word_width = sum(glyphs[index].advance for index in indices)
        if x > left and x + word_width > right:
            next_line()
        for index in indices:
            glyph = glyphs[index]
            if glyph.advance > right - left:
                raise ValueError(
                    "A glyph cannot fit the writing area; reduce size, spacing or margins."
                )
            if x > left and x + glyph.advance > right:
                next_line()
            if page >= MAX_PAGES or (page + 1) * options.width * options.height > MAX_TOTAL_PIXELS:
                raise ValueError("Text exceeds the three-page or seven-million-pixel output limit.")
            y = baseline + glyph.baseline_offset
            if y < options.margin or y + glyph.mask.height > options.height - options.margin:
                raise ValueError("Glyph placement exceeds the vertical writing area.")
            placements.append(
                Placement(
                    index,
                    text[index],
                    page,
                    line,
                    x,
                    y,
                    baseline,
                    glyph.mask.width,
                    glyph.mask.height,
                    glyph.kind,
                )
            )
            x += glyph.advance
    if not placements:
        raise ValueError("No visible supported text to render.")
    # Trailing whitespace alone must not create phantom exported pages.
    return PageLayout(placements, glyphs, max(item.page for item in placements) + 1)


def compose_pages(layout: PageLayout, options: LayoutOptions) -> list[Image.Image]:
    if not 1 <= layout.page_count <= MAX_PAGES:
        raise ValueError("Invalid page count.")
    if layout.page_count * options.width * options.height > MAX_TOTAL_PIXELS:
        raise ValueError("Output pixel budget exceeded.")
    pages = []
    for _ in range(layout.page_count):
        page = Image.new("RGB", (options.width, options.height), "white")
        draw = ImageDraw.Draw(page)
        if options.paper == "ruled":
            for baseline in range(
                options.margin + options.glyph_size,
                options.height - options.margin,
                options.line_step,
            ):
                draw.line(
                    (
                        options.margin // 2,
                        baseline + 3,
                        options.width - options.margin // 2,
                        baseline + 3,
                    ),
                    fill="#DFE8F1",
                    width=1,
                )
            draw.line(
                (
                    options.margin - 18,
                    options.margin // 2,
                    options.margin - 18,
                    options.height - options.margin // 2,
                ),
                fill="#EFD9DC",
                width=1,
            )
        elif options.paper == "grid":
            cell = max(12, options.glyph_size // 2)
            for x in range(options.margin, options.width - options.margin + 1, cell):
                draw.line((x, options.margin, x, options.height - options.margin), fill="#E5ECF2")
            for y in range(options.margin, options.height - options.margin + 1, cell):
                draw.line((options.margin, y, options.width - options.margin, y), fill="#E5ECF2")
        pages.append(page)
    for item in layout.placements:
        glyph = layout.glyphs[item.index]
        pages[item.page].paste(
            (18, 25, 36), (item.x, item.y, item.x + item.width, item.y + item.height), glyph.mask
        )
    return pages
