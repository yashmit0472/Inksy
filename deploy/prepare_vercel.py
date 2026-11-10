"""Package the public studio and its generated glyphs for a standalone Vercel deploy."""

import argparse
import base64
import gzip
import json
from pathlib import Path
from shutil import copy2

from ml.characters import ALPHABET
from ml.print_handwriting import load_letter_variants, with_print_punctuation
from ml.utils import sha256_file

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / "artifacts/print-handwriting-v3-balanced/model.pt"


def prepare(output, checkpoint):
    bank, _ = load_letter_variants(checkpoint)
    if set(bank) != set(ALPHABET):
        raise ValueError("Use the standard mixed-case studio checkpoint.")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("api", "public", "data", "rendering", "ml"):
        (output / name).mkdir()
    for relative in ("api/render.py", "requirements.txt", "vercel.json"):
        copy2(ROOT / "deploy/vercel" / relative, output / relative)
    for relative in ("rendering/print_tiles.py", "rendering/print_pages.py", "ml/characters.py"):
        copy2(ROOT / relative, output / relative)
    for asset in ("index.html", "favicon.svg", "studio.css", "studio.js"):
        copy2(ROOT / "rendering/studio_web" / asset, output / "public" / asset)
    (output / "public/studio-config.json").write_text(
        '{"styles":[],"hosted":true,"training_available":false}\n', encoding="utf-8"
    )
    tiles = {
        char: [(tile.width, base64.b64encode(tile.tobytes()).decode("ascii")) for tile in variants]
        for char, variants in with_print_punctuation(bank).items()
    }
    payload = json.dumps(tiles, separators=(",", ":")).encode()
    (output / "data/glyphs.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    return sha256_file(checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=STANDARD)
    args = parser.parse_args()
    digest = prepare(args.output, args.checkpoint)
    print(f"Prepared {args.output.resolve()} from checkpoint SHA-256 {digest}")


if __name__ == "__main__":
    main()
