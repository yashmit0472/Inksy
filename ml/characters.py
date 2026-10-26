"""Versioned text and label contract. No generated glyphs are replaced by fonts."""

import string
from dataclasses import dataclass

ALPHABET = string.digits + string.ascii_uppercase + string.ascii_lowercase
CHAR_TO_LABEL = {char: index for index, char in enumerate(ALPHABET)}
PUNCTUATION = ".,!?:;'\"-()"
CONTRACT_VERSION = "byclass-v1"
PREPROCESSING_VERSION = "emnist-transpose-foreground-01-v1"
MAX_TEXT_LENGTH = 3_000
NORMALIZATIONS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
}


@dataclass(frozen=True)
class UnsupportedCharacter:
    character: str
    position: int


class UnsupportedText(ValueError):
    def __init__(self, characters: list[UnsupportedCharacter]):
        self.characters = characters
        summary = ", ".join(f"{item.character!r} at {item.position}" for item in characters[:10])
        super().__init__(f"Unsupported characters (zero-based positions): {summary}")


def normalize_text(text: str) -> str:
    """Explicit normalization; callers must show changes before sending a generation request."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    for source, replacement in NORMALIZATIONS.items():
        text = text.replace(source, replacement)
    return text


def validate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> None:
    if not text.strip():
        raise ValueError("Enter at least one supported visible character.")
    if len(text) > max_length:
        raise ValueError(f"Text exceeds the {max_length}-character limit.")
    supported = set(ALPHABET + PUNCTUATION + " \n")
    unsupported = [UnsupportedCharacter(c, i) for i, c in enumerate(text) if c not in supported]
    if unsupported:
        raise UnsupportedText(unsupported)


def encode_characters(text: str) -> list[int]:
    """Encode only model-generated characters; layout/punctuation are handled elsewhere."""
    validate_text(text)
    return [CHAR_TO_LABEL[c] for c in text if c in CHAR_TO_LABEL]
