import random

from PIL import Image

from ml.print_handwriting import compose_varied_word


def test_seeded_repeated_letters_use_both_forms_without_overwriting_ink():
    first = Image.new("L", (5, 48), 255)
    second = Image.new("L", (5, 48), 255)
    first.putpixel((4, 20), 0)
    second.putpixel((0, 24), 64)
    bank = {"a": [first, second]}
    output = compose_varied_word("aa", bank, random.Random(42), {})
    assert output.size == (9, 50)
    assert sorted(value for value in output.tobytes() if value < 255) == [0, 64]
    assert output.tobytes() == compose_varied_word("aa", bank, random.Random(42), {}).tobytes()
