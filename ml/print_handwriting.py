"""Train and render print handwriting from reviewed IAM and EMNIST glyphs."""

import argparse
import json
import random
import string
from pathlib import Path

import torch
from PIL import Image, ImageChops, ImageOps
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from ml.characters import ALPHABET, PUNCTUATION, normalize_text
from ml.iam import IAMWordDataset
from ml.utils import environment_info, seed_everything, sha256_file, write_json
from ml.word_generator_preview import paper_image, save_grid
from rendering.compositor import punctuation_glyph

LETTERS = string.ascii_lowercase
FORMAT = "inksy-iam-print-glyph-v1"
VARIANT_FORMAT = "inksy-iam-print-glyph-v2"
MIXED_FORMAT = "inksy-mixed-print-glyph-v3"
PERSONAL_FORMAT = "inksy-personal-print-v1"
PERSONAL_ALPHANUMERIC_FORMAT = "inksy-personal-print-v2"
INVENTORIES = {
    FORMAT: LETTERS,
    VARIANT_FORMAT: LETTERS,
    MIXED_FORMAT: ALPHABET,
    PERSONAL_FORMAT: LETTERS,
    PERSONAL_ALPHANUMERIC_FORMAT: ALPHABET,
}


class PrintDecoder(nn.Module):
    def __init__(self, glyph_count=26):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Embedding(glyph_count, 48),
            nn.Linear(48, 32 * 6 * 6),
            nn.Unflatten(1, (32, 6, 6)),
            nn.ConvTranspose2d(32, 24, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(24, 16, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(8, 1, 3, padding=1),
        )

    def forward(self, labels):
        return self.layers(labels)


def prepare_glyph(letter, item, source, variants, *, isolated=False):
    if letter not in item["text"]:
        raise ValueError(f"Letter {letter} does not occur in its source label.")
    left, right = source["left"], source["right"]
    if not 0 <= left < right <= item["image"].shape[-1]:
        raise ValueError("Crop bounds are outside the source image.")
    crop = item["image"][0, :, left:right]
    points = (crop > 0.25).nonzero()
    if not len(points):
        raise ValueError(f"Empty letter crop: {letter}")
    top, bottom = int(points[:, 0].min()), int(points[:, 0].max()) + 1
    crop = crop[top:bottom]
    # Suppress the scanned paper and normalize pen darkness using the ink pixels.
    ink = crop[crop > 0.25]
    crop = ((crop - 0.15) / (ink.quantile(0.9) - 0.15)).clamp(0, 1)
    height = 34 if isolated or letter in "bdfhijklt" else 23
    descent = 0
    if letter in "gjpqy":
        height = 42 if letter == "j" else 33
        descent = 9
    width = max(3, min(42, round(crop.shape[1] * height / crop.shape[0])))
    resized = F.interpolate(
        crop[None, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]
    canvas = torch.zeros(1, 48, 48)
    y = 36 + descent - height
    canvas[:, y : y + height, 2 : 2 + width] = resized
    if isolated:
        # Apply more thinning to the wide-stroke outliers among the isolated samples.
        core = canvas > 0.5
        perimeter = core.diff(dim=-1).sum() + core.diff(dim=-2).sum()
        stroke_radius = float(core.sum() / perimeter.clamp_min(1))
        blend = min(0.78, 0.22 + 0.60 * max(0, stroke_radius - 1.25))
        inner = -F.max_pool2d(-canvas, 3, stride=1, padding=1)
        canvas = canvas.lerp(inner, blend)
    elif variants:
        # Reinforce faint neighboring pixels while keeping the original ink geometry.
        neighbor = F.max_pool2d(canvas, 3, stride=1, padding=1)
        canvas = torch.maximum(canvas.pow(0.80), 0.33 * neighbor)
    return canvas, width + 4


def prepare_targets(prepared, selection):
    data = IAMWordDataset(prepared, "train")
    config = json.loads(selection.read_text(encoding="utf-8"))
    if config["records_sha256"] != data.manifest["records_sha256"]:
        raise ValueError("Letter selection belongs to a different dataset.")
    checkpoint_format = config.get("format", FORMAT)
    alphabet = INVENTORIES[checkpoint_format]
    if set(config["glyphs"]) != set(alphabet):
        raise ValueError("The crop selection must cover its complete character inventory.")
    variants = checkpoint_format != FORMAT
    sources = {row["id"]: i for i, row in enumerate(data.rows)}
    if checkpoint_format == MIXED_FORMAT:
        from ml.dataset import load_partitions

        emnist = config["emnist"]
        split_dir = Path(emnist["prepared"])
        if sha256_file(split_dir / "manifest.json") != emnist["manifest_sha256"]:
            raise ValueError("EMNIST manifest differs from the reviewed selection.")
        dataset, train_partition, _, manifest = load_partitions(Path(emnist["root"]), split_dir)
        if manifest["split_sha256"] != emnist["split_sha256"]:
            raise ValueError("EMNIST split differs from the reviewed selection.")
        training_indices = set(train_partition.indices)
    targets, widths, owners = [], [], []
    for letter in alphabet:
        crops = config["glyphs"][letter]
        crops = crops if variants else [crops]
        if not 1 <= len(crops) <= 4:
            raise ValueError("Supply one to four reviewed crops per letter.")
        for crop in crops:
            isolated = checkpoint_format == MIXED_FORMAT and letter not in LETTERS
            if isolated:
                index = crop["emnist_index"]
                if type(index) is not int or index not in training_indices:
                    raise ValueError("An EMNIST sample is outside the project training split.")
                image, label = dataset[index]
                if dataset.classes[label] != letter:
                    raise ValueError("An EMNIST sample does not match its requested class.")
                ink = (image[0] > 0.25).nonzero()
                bounds = {"left": int(ink[:, 1].min()), "right": int(ink[:, 1].max()) + 1}
                item = {"image": image, "text": letter}
            else:
                item, bounds = data[sources[crop["id"]]], crop
            canvas, width = prepare_glyph(letter, item, bounds, variants, isolated=isolated)
            targets.append(canvas)
            widths.append(width)
            owners.append(letter)
    return torch.stack(targets), widths, owners, config


def fit_decoder(targets, updates, seed, *, variants=True, progress=None, model=None):
    if model is None:
        model = PrintDecoder(len(targets))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    labels = torch.arange(len(targets))
    random_order = torch.Generator().manual_seed(seed)
    batches = ()
    history = []
    steps = range(1, updates + 1)
    if progress is None:
        steps = tqdm(steps, desc="Train print letter decoder", ascii=True)
    for update in steps:
        if variants and not batches:
            batches = torch.randperm(len(targets), generator=random_order).split(24)
        if variants:
            selected, *remaining = batches
            batches = tuple(remaining)
        else:
            selected = labels
        batch_targets = targets[selected]
        optimizer.zero_grad(set_to_none=True)
        logits = model(selected)
        bce = (
            F.binary_cross_entropy_with_logits(logits, batch_targets, reduction="none")
            * (1 + 3 * batch_targets)
        ).mean()
        pixels = logits.sigmoid()
        dice = (
            1
            - (2 * (pixels * batch_targets).sum((1, 2, 3)) + 1)
            / ((pixels + batch_targets).sum((1, 2, 3)) + 1)
        ).mean()
        loss = bce + dice
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite glyph loss.")
        loss.backward()
        optimizer.step()
        if update == 1 or update % 100 == 0 or update == updates:
            history.append({"update": update, "objective": float(loss.detach())})
            if progress is not None:
                progress(update)
    return model, history


def train(args):
    source_digest = sha256_file(Path(__file__))
    seed_everything(args.seed)
    targets, widths, owners, selection = prepare_targets(args.prepared, args.selection)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_format = selection.get("format", FORMAT)
    model, history = fit_decoder(
        targets, args.updates, args.seed, variants=checkpoint_format != FORMAT
    )
    checkpoint = args.output / "model.pt"
    temporary = args.output / "model.tmp"
    torch.save(
        {
            "format": checkpoint_format,
            "alphabet": INVENTORIES[checkpoint_format],
            "widths": widths,
            "owners": owners,
            "state_dict": model.state_dict(),
            "seed": args.seed,
            "updates": args.updates,
            "source_records_sha256": selection["records_sha256"],
            "selection_sha256": sha256_file(args.selection),
        },
        temporary,
    )
    temporary.replace(checkpoint)
    write_json(
        args.output / "run.json",
        {
            "checkpoint_sha256": sha256_file(checkpoint),
            "selection": selection,
            "history": history,
            "learning_rate": 0.003,
            "source_sha256": source_digest,
            "environment": environment_info(torch.device("cpu")),
        },
    )
    masks, _ = load_letter_variants(checkpoint)
    save_grid(
        args.output / "alphabet.png",
        [image for images in masks.values() for image in images],
        [f"{letter} {i + 1}" for letter, images in masks.items() for i in range(len(images))],
        "Generated print-style glyphs | reviewed training examples",
    )


@torch.inference_mode()
def load_letter_variants(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    alphabet = INVENTORIES.get(saved["format"])
    if alphabet is None or saved["alphabet"] != alphabet:
        raise ValueError("Incompatible print decoder checkpoint.")
    variants = saved["format"] != FORMAT
    owners = saved["owners"] if variants else list(alphabet)
    if set(owners) != set(alphabet) or len(owners) != len(saved["widths"]):
        raise ValueError("Checkpoint letter inventory is inconsistent.")
    model = PrintDecoder(len(owners))
    model.load_state_dict(saved["state_dict"], strict=True)
    masks = model.eval()(torch.arange(len(owners))).sigmoid()
    output = {letter: [] for letter in alphabet}
    for letter, mask, width in zip(owners, masks, saved["widths"], strict=True):
        image = paper_image(mask[:, :, :width])
        output[letter].append(image)
    return output, variants


def load_letters(checkpoint):
    masks, _ = load_letter_variants(checkpoint)
    return {letter: images[0] for letter, images in masks.items()}


def compose_word(word, letters):
    if not word or len(word) > 64 or any(letter not in letters for letter in word):
        raise ValueError("Use 1–64 lowercase ASCII letters per word.")
    width = sum(letters[letter].width for letter in word)
    image = Image.new("L", (width, 48), 255)
    x = 0
    for letter in word:
        image.paste(letters[letter], (x, 0))
        x += letters[letter].width
    return image


def compose_varied_word(word, letters, rng, previous):
    if not word or len(word) > 64 or any(letter not in letters for letter in word):
        raise ValueError("Use 1–64 supported characters per word.")
    return compose_print_tiles(select_print_tiles(word, letters, rng, previous), rng)


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


def punctuation_tile(mark):
    glyph = punctuation_glyph(mark, 33, 0)
    tile = Image.new("L", (glyph.mask.width + 4, 48), 255)
    tile.paste(ImageOps.invert(glyph.mask), (2, 36 + glyph.baseline_offset))
    return tile


def with_print_punctuation(bank):
    return {**bank, **{mark: [punctuation_tile(mark)] for mark in PUNCTUATION}}


def render(args):
    bank, variants = load_letter_variants(args.checkpoint)
    text = args.text
    if set(bank) == set(ALPHABET):
        text = normalize_text(text)
        bank = with_print_punctuation(bank)
    words = text.split()
    if not words or len(text) > 600:
        raise ValueError("Supply 1–600 characters of text.")
    unsupported = sorted({char for char in text if not char.isspace() and char not in bank})
    if unsupported:
        raise ValueError(f"Characters outside this checkpoint's inventory: {unsupported!r}")
    if variants:
        rng = random.Random(args.seed)
        previous = {}
        lines = [
            [compose_varied_word(word, bank, rng, previous) for word in line.split()]
            for line in text.splitlines()
        ]
    else:
        letters = {letter: images[0] for letter, images in bank.items()}
        lines = [[compose_word(word, letters) for word in words]]
    page = Image.new("L", (920, 62 * (len(words) + len(lines)) + 40), 255)
    x, y = 24, 20
    for line_index, line in enumerate(lines):
        if line_index:
            x, y = 24, y + 62
        for image in line:
            if x + image.width > page.width - 24 and x > 24:
                x, y = 24, y + 62
            if image.width > page.width - 48:
                raise ValueError("A word is too wide to fit on the page.")
            page.paste(image, (x, y))
            x += image.width + (14 if variants else 15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        page.crop((0, 0, page.width, y + 68)).save(stream, format="PNG")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--prepared", type=Path, default=Path("artifacts/iam-words"))
    fit.add_argument("--selection", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--updates", type=int, default=500)
    fit.set_defaults(action=train)
    preview = commands.add_parser("render")
    preview.add_argument("--checkpoint", type=Path, required=True)
    preview.add_argument("--text", required=True)
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--seed", type=int, default=42)
    preview.set_defaults(action=render)
    args = parser.parse_args()
    if hasattr(args, "updates") and not 1 <= args.updates <= 3000:
        parser.error("Use 1–3,000 updates.")
    torch.set_num_threads(4)
    args.action(args)


if __name__ == "__main__":
    main()
