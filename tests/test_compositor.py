import pytest
from PIL import Image, ImageDraw

from ml.characters import CHAR_TO_LABEL, PUNCTUATION
from rendering.compositor import (
    LayoutOptions,
    compose_pages,
    layout_text,
    model_glyph,
    punctuation_glyph,
)


def fixture_mask(width=12, height=22):
    mask = Image.new("L", (28, 28))
    left, top = (28 - width) // 2, (28 - height) // 2
    ImageDraw.Draw(mask).rectangle((left, top, left + width - 1, top + height - 1), fill=255)
    return mask


def fixture_masks(text):
    return {index: fixture_mask() for index, char in enumerate(text) if char in CHAR_TO_LABEL}


def test_proportional_widths_and_descenders_are_not_square_boxes():
    options = LayoutOptions()
    narrow = model_glyph("i", fixture_mask(2, 22), options)
    wide = model_glyph("W", fixture_mask(22, 20), options)
    descender = model_glyph("g", fixture_mask(), options)
    capital = model_glyph("A", fixture_mask(), options)
    assert narrow.advance < wide.advance
    assert descender.baseline_offset + descender.mask.height > 0
    assert capital.baseline_offset + capital.mask.height == 0


def test_wrapping_preserves_visible_characters_in_order_and_keeps_fitting_words_together():
    text = "Alpha Beta Gamma Delta " + "Ab" * 35
    options = LayoutOptions(width=320, height=600, margin=24, glyph_size=20)
    layout = layout_text(text, fixture_masks(text), options)
    assert [(p.index, p.character) for p in layout.placements] == [
        (index, char) for index, char in enumerate(text) if char not in " \n"
    ]
    for word in ("Alpha", "Beta", "Gamma", "Delta"):
        start = text.index(word)
        positions = [p for p in layout.placements if start <= p.index < start + len(word)]
        assert len({(p.page, p.line) for p in positions}) == 1
    assert len({(p.page, p.line) for p in layout.placements}) > 1
    assert all(
        options.margin <= p.x and p.x + p.width <= options.width - options.margin
        for p in layout.placements
    )
    assert all(
        options.margin <= p.y and p.y + p.height <= options.height - options.margin
        for p in layout.placements
    )


def test_spaces_and_blank_lines_have_explicit_geometry():
    text = "  A\n\nB"
    options = LayoutOptions()
    layout = layout_text(text, fixture_masks(text), options)
    first, second = layout.placements
    assert first.x == options.margin + 2 * options.space_advance
    assert second.baseline - first.baseline == 2 * options.line_step


def test_page_breaks_and_trailing_newlines_do_not_create_phantom_pages():
    options = LayoutOptions(width=320, height=320, margin=24, glyph_size=24, line_spacing=12)
    text = "A\n" * 12
    layout = layout_text(text, fixture_masks(text), options)
    assert layout.page_count == 2
    assert len(compose_pages(layout, options)) == 2
    trailing = "A" + "\n" * 100
    assert layout_text(trailing, fixture_masks(trailing), options).page_count == 1


def test_too_many_pages_fail_before_page_images_are_allocated():
    text = "A\n" * 50
    options = LayoutOptions(width=320, height=320, margin=24, glyph_size=24)
    with pytest.raises(ValueError, match="output limit"):
        layout_text(text, fixture_masks(text), options)


def test_total_pixel_limit_is_enforced_across_pages():
    options = LayoutOptions(width=2048, height=2048)
    text = "A" + "\n" * 50 + "B"
    with pytest.raises(ValueError, match="output limit"):
        layout_text(text, fixture_masks(text), options)


def test_missing_model_glyph_is_an_error_not_a_font_fallback():
    with pytest.raises(ValueError, match="Missing generated glyph"):
        layout_text("Hello", {}, LayoutOptions())


def test_punctuation_only_is_explicitly_procedural():
    layout = layout_text(PUNCTUATION, {}, LayoutOptions())
    assert all(item.kind == "procedural" for item in layout.placements)
    assert len(layout.placements) == len(PUNCTUATION)
    assert all(punctuation_glyph(char, 32, 2).mask.getbbox() for char in PUNCTUATION)


def test_mask_alpha_does_not_paint_a_white_rectangle_over_paper():
    options = LayoutOptions(paper="plain")
    layout = layout_text("A", fixture_masks("A"), options)
    page = compose_pages(layout, options)[0]
    p = layout.placements[0]
    assert page.getpixel((p.x + p.width // 2, p.y + p.height // 2)) == (18, 25, 36)
    assert page.getpixel((p.x - 1, p.y)) == (255, 255, 255)


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 100},
        {"height": 3000},
        {"glyph_size": 0},
        {"margin": 700},
        {"character_spacing": -1},
        {"word_spacing": -1},
        {"line_spacing": -1},
        {"paper": "unknown"},
        {"glyph_size": 20.5},
    ],
)
def test_invalid_layout_is_rejected(changes):
    with pytest.raises(ValueError):
        LayoutOptions(**changes)
