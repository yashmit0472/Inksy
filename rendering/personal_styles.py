"""Validate local worksheet samples and train a separate print checkpoint."""

import base64
import binascii
import hashlib
import io
import json
import logging
import re
import string
import threading
import uuid
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.transforms.functional import pil_to_tensor

from ml.characters import ALPHABET
from ml.print_handwriting import (
    LETTERS,
    PERSONAL_ALPHANUMERIC_FORMAT,
    PERSONAL_FORMAT,
    PrintDecoder,
    fit_decoder,
    load_letter_variants,
    prepare_glyph,
    with_print_punctuation,
)
from ml.utils import sha256_file, write_json

SHEET_SIZE = (96, 96)
UPDATES = 900
SEED = 42
OWNER_LABELS = tuple(letter for letter in LETTERS for _ in range(2))
WORKSHEET_ALPHABET = LETTERS + string.ascii_uppercase + string.digits
FULL_OWNER_LABELS = tuple(letter for letter in WORKSHEET_ALPHABET for _ in range(2))
MAX_SAMPLE_LENGTH = 50_000


def worksheet_targets(samples, owners=OWNER_LABELS):
    if not isinstance(samples, list) or len(samples) != len(owners):
        raise ValueError(f"Complete both boxes for all {len(owners) // 2} characters.")
    targets, widths, hashes = [], [], []
    for index, (letter, encoded) in enumerate(zip(owners, samples, strict=True)):
        label = f"{letter}, sample {index % 2 + 1}"
        try:
            if not isinstance(encoded, str) or len(encoded) > MAX_SAMPLE_LENGTH:
                raise ValueError("Expected a PNG.")
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as source:
                if source.format != "PNG" or source.size != SHEET_SIZE:
                    raise ValueError("Unexpected worksheet image.")
                rgba = source.convert("RGBA")
        except (
            ValueError,
            binascii.Error,
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
        ) as error:
            raise ValueError(f"Redraw {label}. Its image could not be read.") from error
        white = Image.new("RGBA", SHEET_SIZE, "white")
        white.alpha_composite(rgba)
        image = pil_to_tensor(ImageOps.invert(white.convert("L"))).float().div(255)
        ink = (image[0] > 0.25).nonzero()
        if len(ink) < 18 or int(ink[:, 0].max() - ink[:, 0].min()) < 8:
            raise ValueError(f"Write a full character in the box for {label}.")
        if len(ink) > 4000 or int(ink.min()) == 0 or int(ink.max()) == SHEET_SIZE[0] - 1:
            raise ValueError(f"Keep the character inside the box for {label}.")
        bounds = {"left": int(ink[:, 1].min()), "right": int(ink[:, 1].max()) + 1}
        target, width = prepare_glyph(
            letter, {"image": image, "text": letter}, bounds, True, isolated=letter not in LETTERS
        )
        targets.append(target)
        widths.append(width)
        hashes.append(hashlib.sha256(raw).hexdigest())
    return torch.stack(targets), widths, hashes


class PersonalStyles:
    def __init__(self, root, *, updates=UPDATES):
        self.root = Path(root).resolve()
        self.updates = updates
        self.lock = threading.Lock()
        self.state = {"phase": "idle"}
        self.cache = {}

    def list(self):
        if not self.root.exists():
            return []
        styles = []
        for manifest in sorted(self.root.glob("*/style.json")):
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            alphabet = (
                ALPHABET if metadata.get("format") == PERSONAL_ALPHANUMERIC_FORMAT else LETTERS
            )
            styles.append(
                {"id": manifest.parent.name, "name": metadata["name"], "alphabet": alphabet}
            )
        return styles

    def bank(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("Choose an available handwriting style.")
        with self.lock:
            if identifier in self.cache:
                return self.cache[identifier]
            directory = self.root / identifier
            if not (directory / "style.json").is_file():
                raise ValueError("That handwriting style is unavailable.")
            bank, _ = load_letter_variants(directory / "model.pt")
            self.cache[identifier] = with_print_punctuation(bank)
            return self.cache[identifier]

    def status(self):
        with self.lock:
            return self.state.copy()

    def start(self, name, samples, *, full=False, photo=False):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 40:
            raise ValueError("Enter a style name of 1 to 40 characters.")
        owners = FULL_OWNER_LABELS if full else OWNER_LABELS
        targets, widths, hashes = worksheet_targets(samples, owners)
        identifier = uuid.uuid4().hex
        with self.lock:
            if self.state["phase"] == "training":
                raise ValueError("A style is already training. Wait for it to finish.")
            self.state = {"phase": "training", "step": 0, "total": self.updates}
        job = threading.Thread(
            target=self._fit,
            args=(identifier, name.strip(), targets, widths, hashes, owners, photo),
            daemon=True,
        )
        job.start()

    def _fit(self, identifier, name, targets, widths, hashes, owners, photo):
        def progress(step):
            with self.lock:
                self.state["step"] = step

        try:
            full = owners == FULL_OWNER_LABELS
            checkpoint_format = PERSONAL_ALPHANUMERIC_FORMAT if full else PERSONAL_FORMAT
            alphabet = ALPHABET if full else LETTERS
            with self.lock, torch.random.fork_rng(devices=[]):
                torch.manual_seed(SEED)
                model = PrintDecoder(len(targets))
            model, history = fit_decoder(
                targets, self.updates, SEED, progress=progress, model=model
            )
            directory = self.root / identifier
            directory.mkdir(parents=True, exist_ok=False)
            checkpoint = directory / "model.pt"
            torch.save(
                {
                    "format": checkpoint_format,
                    "alphabet": alphabet,
                    "owners": list(owners),
                    "widths": widths,
                    "state_dict": model.state_dict(),
                    "seed": SEED,
                    "updates": self.updates,
                },
                checkpoint,
            )
            with self.lock:
                bank, _ = load_letter_variants(checkpoint)
            metadata = directory / "style.tmp"
            write_json(
                metadata,
                {
                    "name": name,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "format": checkpoint_format,
                    "alphabet": alphabet,
                    "seed": SEED,
                    "updates": self.updates,
                    "learning_rate": 0.003,
                    "trainer_sha256": sha256_file(Path(__file__)),
                    "source": "photo worksheet with optional drawing corrections"
                    if photo
                    else "local drawing worksheet",
                    "sample_sha256": dict(
                        zip(
                            (f"{letter}-{i % 2 + 1}" for i, letter in enumerate(owners)),
                            hashes,
                            strict=True,
                        )
                    ),
                    "history": history,
                },
            )
            metadata.replace(directory / "style.json")
            with self.lock:
                self.cache[identifier] = with_print_punctuation(bank)
                self.state = {"phase": "done", "style": identifier, "name": name}
        except Exception:
            logging.exception("Personal style training failed")
            with self.lock:
                self.state = {
                    "phase": "error",
                    "error": "Training stopped. Try again from your worksheet.",
                }
