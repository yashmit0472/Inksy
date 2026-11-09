from PIL import Image, ImageChops

from rendering.print_pages import LINE_STEP, MARGIN, PAGE_SIZE, print_pages


def test_long_word_wraps_without_losing_ink():
    tile = Image.new("L", (34, 48), 255)
    tile.putpixel((5, 20), 0)
    pages = print_pages("A" * 100, {"A": [tile]})
    assert len(pages) == 1
    assert pages[0].size == PAGE_SIZE
    assert pages[0].tobytes().count(0) == 100
    left, top, right, bottom = ImageChops.invert(pages[0]).getbbox()
    assert left >= MARGIN and top >= MARGIN
    assert right <= PAGE_SIZE[0] - MARGIN and bottom <= PAGE_SIZE[1] - MARGIN


def test_paragraph_breaks_and_pagination_preserve_rows():
    tile = Image.new("L", (10, 48), 255)
    tile.putpixel((4, 20), 0)
    pages = print_pages("A\n\n" + "\n".join(["A"] * 20), {"A": [tile]})
    assert len(pages) == 2
    assert sum(page.tobytes().count(0) for page in pages) == 21
    rows = [y for y in range(PAGE_SIZE[1]) if pages[0].getpixel((MARGIN + 4, y)) == 0]
    assert abs(rows[1] - rows[0] - 2 * LINE_STEP) <= 2
