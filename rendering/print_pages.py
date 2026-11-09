"""Lay out learned print glyphs on fixed-size, exportable pages."""

import random

from PIL import Image

from ml.characters import normalize_text
from ml.print_handwriting import compose_print_tiles, select_print_tiles

PAGE_SIZE = (920, 1190)
MARGIN = 64
LINE_STEP = 62
GLYPH_HEIGHT = 50
WORD_GAP = 14
MAX_TEXT = 6000
MAX_PAGES = 12


def split_tiles(tiles, width):
    """Split a word only when its full image exceeds the printable width."""
    group = []
    used = 1
    for tile in tiles:
        advance = tile.width - 1
        if group and used + advance > width:
            yield group, used
            group, used = [], 1
        group.append(tile)
        used += advance
    if group:
        yield group, used


def print_pages(text, bank, seed=42):
    text = normalize_text(text)
    if not text.strip() or len(text) > MAX_TEXT:
        raise ValueError(f"Enter between 1 and {MAX_TEXT:,} characters.")
    unsupported = sorted({char for char in text if not char.isspace() and char not in bank})
    if unsupported:
        sample = ", ".join(repr(char) for char in unsupported[:8])
        raise ValueError(f"Unsupported characters in your text: {sample}.")

    pages = [Image.new("L", PAGE_SIZE, 255)]
    x = y = MARGIN
    right = PAGE_SIZE[0] - MARGIN
    bottom = PAGE_SIZE[1] - MARGIN
    rng, previous = random.Random(seed), {}

    def next_line():
        nonlocal x, y
        x, y = MARGIN, y + LINE_STEP
        if y + GLYPH_HEIGHT > bottom:
            if len(pages) >= MAX_PAGES:
                raise ValueError(
                    f"This text takes more than {MAX_PAGES} pages. Shorten it to continue."
                )
            pages.append(Image.new("L", PAGE_SIZE, 255))
            y = MARGIN

    for line_index, line in enumerate(text.splitlines()):
        if line_index:
            next_line()
        for word in line.split():
            tiles = select_print_tiles(word, bank, rng, previous)
            for chunk_index, (chunk, width) in enumerate(split_tiles(tiles, right - MARGIN)):
                if chunk_index or x + width > right:
                    next_line()
                pages[-1].paste(compose_print_tiles(chunk, rng), (x, y))
                x += width + WORD_GAP
    return pages
