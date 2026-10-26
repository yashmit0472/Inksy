import pytest

from ml.characters import (
    ALPHABET,
    UnsupportedText,
    encode_characters,
    normalize_text,
    validate_text,
)


def test_byclass_identity_and_case_are_preserved():
    assert len(ALPHABET) == len(set(ALPHABET)) == 62
    assert encode_characters("0AaZz9") == [0, 10, 36, 35, 61, 9]


def test_layout_and_punctuation_are_not_model_classes():
    assert encode_characters("Hi,\n 7!") == [17, 44, 7]


def test_normalization_is_explicit_and_idempotent():
    raw = "\u201cHi\u201d\u2014it\u2019s\u2026\r\n\tOK\r"
    result = '"Hi"-it\'s...\n    OK\n'
    assert normalize_text(raw) == result
    assert normalize_text(result) == result
    with pytest.raises(UnsupportedText):
        validate_text(raw)
    validate_text(result)


@pytest.mark.parametrize("text", ["", " ", "\n\n"])
def test_empty_input_is_rejected(text):
    with pytest.raises(ValueError, match="visible"):
        validate_text(text)


def test_unicode_and_controls_report_positions_without_dropping_text():
    with pytest.raises(UnsupportedText) as error:
        validate_text("a\u00e9\x00")
    assert [(c.character, c.position) for c in error.value.characters] == [
        ("\u00e9", 1),
        ("\x00", 2),
    ]


def test_size_is_bounded_after_normalization():
    with pytest.raises(ValueError, match="limit"):
        validate_text(normalize_text("\u2026" * 1001))
