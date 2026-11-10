"""Select and lay out learned glyph tiles without importing the training runtime."""

from PIL import Image, ImageChops


def select_print_tiles(word, letters, rng, previous):
    tiles = []
    for letter in word:
        choices = letters[letter]
        index = rng.choice([i for i in range(len(choices)) if i != previous.get(letter)] or [0])
        previous[letter] = index
        tiles.append(choices[index])
    return tiles


def compose_print_tiles(tiles, rng):
    # A small overlap removes duplicated side bearings. Dark strokes survive the overlap.
    output = Image.new("L", (sum(tile.width for tile in tiles) - len(tiles) + 1, 50), 255)
    x = 0
    for tile in tiles:
        y = rng.randrange(3)
        bounds = (x, y, x + tile.width, y + tile.height)
        output.paste(ImageChops.darker(output.crop(bounds), tile), (x, y))
        x += tile.width - 1
    return output
