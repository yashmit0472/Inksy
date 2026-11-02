"""IAM word metadata, leakage-aware preparation and variable-width non-training loader."""

import argparse
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, UnidentifiedImageError
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from ml.characters import ALPHABET, PUNCTUATION
from ml.utils import sha256_file, write_json

WORD_ALPHABET = ALPHABET + PUNCTUATION
WORD_TOKENS = {char: index + 1 for index, char in enumerate(WORD_ALPHABET)}
WORD_CONTRACT = "iam-word-ctc-blank0-v1"
WORD_PREPROCESSING = "iam-upright-height64-margin4-aspect-foreground-v1"
TRAINABLE_PARTITIONS = {"train", "seen_validation", "content_validation"}
SPLIT_FILES = {
    "official_train": "trainset.txt",
    "writer_validation1": "validationset1.txt",
    "writer_validation2": "validationset2.txt",
    "test": "testset.txt",
}
FORM_ID = re.compile(r"[a-z]\d{2}-\d{3}[a-z]?")
WORD_ID = re.compile(r"[a-z]\d{2}-\d{3}[a-z]?-[a-z]?\d{2}-\d{2}")


@dataclass(frozen=True)
class IAMConfig:
    height: int = 64
    margin: int = 4
    max_width: int = 384
    max_characters: int = 24
    seed: int = 42
    content_validation_fraction: float = 0.1
    seen_validation_fraction: float = 0.1

    def __post_init__(self):
        if self.height != 64 or self.margin != 4:
            raise ValueError("This preprocessing contract uses height 64 and margin 4.")
        if not 64 <= self.max_width <= 1024 or self.max_width % 8:
            raise ValueError("Maximum width must be 64-1024 and divisible by eight.")
        if not 1 <= self.max_characters <= 64:
            raise ValueError("Maximum character count must be 1-64.")
        if not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be in [0, 2^32).")
        if not all(
            0 < value < 0.5
            for value in (
                self.content_validation_fraction,
                self.seen_validation_fraction,
            )
        ):
            raise ValueError("Holdout fractions must be between zero and one half.")


def find_unique(root: Path, name: str) -> Path:
    paths = sorted(root.rglob(name))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {name} under {root}; found {len(paths)}.")
    return paths[0]


def data_lines(path: Path):
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip() and not line.lstrip().startswith("#"):
            yield number, line.strip()


def read_forms(path: Path) -> dict[str, str]:
    result = {}
    for number, line in data_lines(path):
        fields = line.split()
        if len(fields) < 2 or not FORM_ID.fullmatch(fields[0]) or not fields[1].isdigit():
            raise ValueError(f"Malformed IAM forms row at {path}:{number}.")
        form, writer = fields[:2]
        if form in result:
            raise ValueError(f"Duplicate IAM form: {form}")
        result[form] = writer
    if not result:
        raise ValueError("IAM form metadata is empty.")
    return result


def read_words(path: Path, forms: dict[str, str]) -> list[dict]:
    result, seen = [], set()
    for number, line in data_lines(path):
        # Actual IAM data has nine fields; its comment header mentions an extra component field.
        fields = line.split(maxsplit=8)
        if len(fields) != 9 or not WORD_ID.fullmatch(fields[0]):
            raise ValueError(f"Malformed IAM words row at {path}:{number}.")
        identifier, status, _, x, y, width, height, _, text = fields
        form = "-".join(identifier.split("-")[:2])
        if identifier in seen or form not in forms:
            raise ValueError(f"Duplicate word or unknown form: {identifier}")
        if status not in {"ok", "err", "er"}:
            raise ValueError(f"Unknown IAM segmentation status: {status}")
        try:
            box = [int(value) for value in (x, y, width, height)]
        except ValueError as error:
            raise ValueError(f"Invalid IAM bounding box for {identifier}.") from error
        seen.add(identifier)
        result.append(
            {
                "id": identifier,
                "form_id": form,
                "line_id": identifier.rsplit("-", 1)[0],
                "writer_id": forms[form],
                "text": text,
                "segmentation": status,
                "bounding_box": box,
            }
        )
    if not result:
        raise ValueError("IAM word metadata is empty.")
    return result


def read_splits(root: Path, forms: dict[str, str]):
    lines, writers, paths = {}, defaultdict(set), {}
    for partition, filename in SPLIT_FILES.items():
        path = find_unique(root, filename)
        paths[partition] = path
        for _, line in data_lines(path):
            form = line.rsplit("-", 1)[0]
            if form not in forms or line in lines:
                raise ValueError(f"Unknown form or overlapping official IAM split line: {line}")
            lines[line] = partition
            writers[partition].add(forms[form])
    for name, group in writers.items():
        for other, other_group in writers.items():
            if name != other and group & other_group:
                raise ValueError(f"Official IAM writer overlap: {name}, {other}")
    if any(not writers[name] for name in SPLIT_FILES):
        raise ValueError("An official IAM partition is empty.")
    return lines, writers, paths


def hash_fraction(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_partition(row: dict, lines: dict, config: IAMConfig) -> str:
    partition = lines.get(row["line_id"], "unused")
    if partition != "official_train":
        return partition
    if hash_fraction(config.seed, "vocabulary:" + row["text"].casefold()) < (
        config.content_validation_fraction
    ):
        return "content_validation"
    if hash_fraction(config.seed, "line:" + row["line_id"]) < config.seen_validation_fraction:
        return "seen_validation"
    return "train"


def ensure_training_styles(rows: list[dict], seed: int):
    """Keep one eligible line for rare training writers; never move held-out word content."""
    grouped = defaultdict(list)
    for row in rows:
        if row["partition"] in TRAINABLE_PARTITIONS:
            grouped[row["writer_id"]].append(row)
    promotions = []
    for writer, group in sorted(grouped.items()):
        if any(row["partition"] == "train" for row in group):
            continue
        candidates = {row["line_id"] for row in group if row["partition"] == "seen_validation"}
        if not candidates:
            raise ValueError(
                f"Writer {writer} has no training line outside the held-out vocabulary. "
                "Inspect the split configuration; content-validation words were not moved."
            )
        chosen = min(
            candidates, key=lambda line: (hash_fraction(seed, "style-anchor:" + line), line)
        )
        count = 0
        for row in group:
            if row["line_id"] == chosen and row["partition"] == "seen_validation":
                row["partition"] = "train"
                count += 1
        promotions.append({"writer_id": writer, "line_id": chosen, "words": count})
    return promotions


def encode_word(text: str) -> torch.Tensor:
    if not text or any(char not in WORD_TOKENS for char in text):
        raise ValueError("Word contains unsupported characters or whitespace.")
    return torch.tensor([WORD_TOKENS[char] for char in text], dtype=torch.long)


def ctc_min_steps(text: str) -> int:
    return len(text) + sum(left == right for left, right in zip(text, text[1:], strict=False))


def normalized_width(size: tuple[int, int], config: IAMConfig):
    width, height = size
    content_width = max(1, round(width * (config.height - 2 * config.margin) / height))
    return math.ceil((content_width + 2 * config.margin) / 8) * 8


def preprocess_word(image: Image.Image, config: IAMConfig):
    if min(image.size) < 1 or image.width * image.height > 4_000_000:
        raise ValueError("Invalid or excessive IAM image dimensions.")
    width = normalized_width(image.size, config)
    if width > config.max_width:
        raise ValueError("Word is too wide; cropping or aspect distortion is not allowed.")
    content_height = config.height - 2 * config.margin
    content_width = max(1, round(image.width * content_height / image.height))
    resized = image.convert("L").resize((content_width, content_height), Image.Resampling.LANCZOS)
    ink = 1 - torch.from_numpy(np.asarray(resized).copy()).float().div(255)
    canvas = torch.zeros((1, config.height, width))
    canvas[
        :,
        config.margin : config.margin + content_height,
        config.margin : config.margin + content_width,
    ] = ink
    return canvas


def image_path(root: Path, identifier: str):
    parts = identifier.split("-")
    relative = Path(parts[0]) / "-".join(parts[:2]) / f"{identifier}.png"
    candidates = [root / "words" / relative, root / "words" / "words" / relative]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one source image for {identifier}.")
    return found[0]


def sample_sheet(rows, root, config, output):
    grouped = defaultdict(list)
    for row in rows:
        if row["partition"] == "train":
            grouped[row["writer_id"]].append(row)
    selected = [grouped[writer][:4] for writer in sorted(grouped)[:8]]
    sheet = Image.new("RGB", (4 * 410, max(1, len(selected)) * 108 + 32), "#edf2f7")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 8),
        "REAL IAM training references; labels are metadata, not generated handwriting.",
        fill="#17283e",
    )
    for row_index, group in enumerate(selected):
        for column, row in enumerate(group):
            with Image.open(root / row["image"]) as source:
                pixels = preprocess_word(source, config)
            image = Image.fromarray((255 * (1 - pixels[0])).round().byte().numpy())
            x, y = column * 410 + 8, row_index * 108 + 36
            sheet.paste(image.convert("RGB"), (x, y))
            draw.text((x, y + 68), f"writer {row['writer_id']}: {row['text']}", fill="#17283e")
    sheet.save(output)


def prepare(root: Path, output: Path, config: IAMConfig | None = None):
    config = config or IAMConfig()
    root = root.resolve()
    if output.exists():
        raise FileExistsError("Choose a new IAM prepared directory; existing runs are preserved.")
    forms_path, words_path = (
        find_unique(root / "ascii", "forms.txt"),
        find_unique(root / "ascii", "words.txt"),
    )
    forms = read_forms(forms_path)
    source_rows = read_words(words_path, forms)
    lines, official_writers, split_paths = read_splits(root / "splits", forms)
    rows, excluded = [], []
    for row in tqdm(source_rows, desc="Check IAM words", unit="word", ascii=True):
        text = row["text"]
        partition = assign_partition(row, lines, config)
        reason = None
        if partition == "unused":
            reason = "outside_official_task"
        elif row["segmentation"] != "ok":
            reason = "segmentation_error"
        elif min(row["bounding_box"][:2]) < 0 or min(row["bounding_box"][2:]) < 1:
            reason = "invalid_bounding_box"
        elif len(text) > config.max_characters:
            reason = "text_too_long"
        elif any(char not in WORD_TOKENS for char in text):
            reason = "unsupported_characters"
        elif not any(char in ALPHABET for char in text):
            reason = "punctuation_only"
        if reason:
            excluded.append({"id": row["id"], "text": text, "reason": reason})
            continue
        # Reserve official unknown-writer validation and final test pixels unopened.
        processed = {
            **row,
            "partition": partition,
            "image": None,
            "image_sha256": None,
            "source_size": None,
            "width": None,
        }
        if partition in TRAINABLE_PARTITIONS:
            path = image_path(root, row["id"])
            raw = path.read_bytes()
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    size = image.size
                    if min(size) < 1 or size[0] * size[1] > 4_000_000:
                        reason = "invalid_image_dimensions"
                    elif normalized_width(size, config) > config.max_width:
                        reason = "image_too_wide"
                    else:
                        width = normalized_width(size, config)
                        if width // 4 < ctc_min_steps(text):
                            reason = "ctc_alignment_too_short"
                        else:
                            image.load()
            except (UnidentifiedImageError, OSError, ValueError):
                reason = "corrupt_image"
            if reason:
                excluded.append({"id": row["id"], "text": text, "reason": reason})
                continue
            processed.update(
                {
                    "image": path.relative_to(root).as_posix(),
                    "image_sha256": hashlib.sha256(raw).hexdigest(),
                    "source_size": list(size),
                    "width": width,
                }
            )
        rows.append(processed)
    style_anchors = ensure_training_styles(rows, config.seed)
    training = [row for row in rows if row["partition"] == "train"]
    if not training:
        raise ValueError("No usable IAM training words survived preparation.")
    writers = {
        writer: index for index, writer in enumerate(sorted({row["writer_id"] for row in training}))
    }
    for row in rows:
        row["writer_index"] = writers.get(row["writer_id"], -1)
    training_vocabulary = {row["text"].casefold() for row in training}
    heldout_vocabulary = {
        row["text"].casefold() for row in rows if row["partition"] == "content_validation"
    }
    if training_vocabulary & heldout_vocabulary:
        raise ValueError("Content-validation vocabulary leaked into training.")
    output.mkdir(parents=True, exist_ok=False)
    for filename, data in (("records.jsonl", rows), ("excluded.jsonl", excluded)):
        with (output / filename).open("w", encoding="utf-8", newline="\n") as stream:
            for row in data:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    writer_word_counts = Counter(row["writer_id"] for row in training)
    sanity_writer = sorted(
        writer_word_counts, key=lambda writer: (-writer_word_counts[writer], writer)
    )[0]
    sanity, sanity_texts = [], set()
    for row in sorted(training, key=lambda item: item["id"]):
        if (
            row["writer_id"] == sanity_writer
            and 2 <= len(row["text"]) <= 12
            and row["text"].casefold() not in sanity_texts
        ):
            sanity.append(row["id"])
            sanity_texts.add(row["text"].casefold())
        if len(sanity) == 16:
            break
    write_json(
        output / "sanity-set.json",
        {
            "writer_id": sanity_writer,
            "word_ids": sanity,
            "partition": "train",
            "purpose": "Future owner-run memorization sanity test only; not a quality evaluation.",
            "training_performed": False,
        },
    )
    character_counts = Counter("".join(row["text"] for row in training))
    manifest = {
        "dataset": "IAM Handwriting Database 3.0",
        "source": "https://zenodo.org/records/19003721",
        "terms": "Non-commercial research purposes; cite Marti and Bunke (2002).",
        "contract": WORD_CONTRACT,
        "preprocessing": WORD_PREPROCESSING,
        "alphabet": WORD_ALPHABET,
        "blank_and_padding_token": 0,
        "root": str(root),
        "config": asdict(config),
        "source_word_rows": len(source_rows),
        "source_sha256": {
            "forms.txt": sha256_file(forms_path),
            "words.txt": sha256_file(words_path),
            **{name: sha256_file(path) for name, path in split_paths.items()},
        },
        "records_sha256": sha256_file(output / "records.jsonl"),
        "exclusions_sha256": sha256_file(output / "excluded.jsonl"),
        "partition_counts": dict(Counter(row["partition"] for row in rows)),
        "official_writer_counts": {name: len(group) for name, group in official_writers.items()},
        "training_writer_to_index": writers,
        "seen_validation_lines_promoted_for_style_coverage": style_anchors,
        "training_words_per_writer": dict(sorted(writer_word_counts.items())),
        "sanity_set_sha256": sha256_file(output / "sanity-set.json"),
        "training_character_counts": {char: character_counts[char] for char in WORD_ALPHABET},
        "characters_absent_from_training": [
            char for char in WORD_ALPHABET if not character_counts[char]
        ],
        "exclusion_counts": dict(Counter(row["reason"] for row in excluded)),
        "known_writer_validation_missing_training_style": sum(
            row["writer_index"] < 0 for row in rows if row["partition"] in TRAINABLE_PARTITIONS
        ),
        "content_validation_casefold_overlap_with_train": 0,
        "test_pixels_opened": False,
        "unknown_writer_validation_pixels_opened": False,
        "training_performed": False,
        "scope": (
            "Official line splits retained; extra lexicon and line holdouts inside official train."
        ),
    }
    write_json(output / "manifest.json", manifest)
    sample_sheet(rows, root, config, output / "real-training-words.png")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "partition_counts",
                    "exclusion_counts",
                    "characters_absent_from_training",
                )
            },
            indent=2,
        )
    )
    return manifest


class IAMWordDataset(Dataset):
    """Known-writer development data; reserved partitions need a future explicit evaluator."""

    def __init__(self, prepared: Path, partition="train"):
        if partition not in TRAINABLE_PARTITIONS:
            raise ValueError(
                "Reserved IAM writer-validation/test images are not opened by this loader."
            )
        self.manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
        if (
            self.manifest["contract"] != WORD_CONTRACT
            or self.manifest["preprocessing"] != WORD_PREPROCESSING
            or self.manifest["alphabet"] != WORD_ALPHABET
        ):
            raise ValueError("Incompatible IAM preprocessing or token contract.")
        path = prepared / "records.jsonl"
        if sha256_file(path) != self.manifest["records_sha256"]:
            raise ValueError("Prepared IAM records changed.")
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.rows = [row for row in self.rows if row["partition"] == partition]
        self.root = Path(self.manifest["root"])
        self.config = IAMConfig(**self.manifest["config"])
        if not self.rows:
            raise ValueError(f"IAM partition is empty: {partition}")
        if any(row["writer_index"] < 0 for row in self.rows):
            raise ValueError("A development writer lacks a training style; inspect the manifest.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["image"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Prepared IAM image path escapes the data root.")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
            raise ValueError(f"IAM source image changed: {row['id']}")
        with Image.open(io.BytesIO(raw)) as image:
            pixels = preprocess_word(image, self.config)
        return {
            "image": pixels,
            "tokens": encode_word(row["text"]),
            "text": row["text"],
            "writer_index": row["writer_index"],
            "id": row["id"],
        }


def collate_words(items):
    if not items:
        raise ValueError("Cannot collate an empty word batch.")
    widths = torch.tensor([item["image"].shape[-1] for item in items], dtype=torch.long)
    images = torch.stack(
        [F.pad(item["image"], (0, int(widths.max()) - item["image"].shape[-1])) for item in items]
    )
    tokens = torch.nn.utils.rnn.pad_sequence([item["tokens"] for item in items], batch_first=True)
    return {
        "images": images,
        "widths": widths,
        "tokens": tokens,
        "token_lengths": torch.tensor([len(item["tokens"]) for item in items], dtype=torch.long),
        "ctc_targets": torch.cat([item["tokens"] for item in items]),
        "writer_indices": torch.tensor([item["writer_index"] for item in items], dtype=torch.long),
        "text": [item["text"] for item in items],
        "ids": [item["id"] for item in items],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/iam"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/iam-words"))
    parser.add_argument("--config", type=Path, default=Path("configs/iam.json"))
    args = parser.parse_args()
    prepare(
        args.root, args.output, IAMConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    )


if __name__ == "__main__":
    main()
