"""Generate requested character variations from a local trained checkpoint."""

import argparse
from pathlib import Path

import torch

from ml.characters import ALPHABET, CHAR_TO_LABEL
from ml.checkpoint import load_checkpoint
from ml.utils import resolve_device, seed_everything, sha256_file, write_json
from ml.visualize import save_grid


@torch.inference_mode()
def sample_glyphs(model, characters: str, seed: int, variations=1):
    if not characters or any(char not in CHAR_TO_LABEL for char in characters):
        raise ValueError("Glyph sampling accepts only A-Z, a-z and 0-9.")
    if not 1 <= variations <= 16:
        raise ValueError("Variations must be between one and sixteen.")
    if len(characters) * variations > 4096:
        raise ValueError("Glyph grid is limited to 4,096 images.")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    labels = torch.tensor([CHAR_TO_LABEL[c] for c in characters], device=device).repeat_interleave(
        variations
    )
    latent = torch.randn((len(labels), model.config.latent_dim), device=device, generator=generator)
    return model.decode(latent, labels).sigmoid().cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--characters", default=ALPHABET)
    parser.add_argument("--variations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/generated/glyphs.png"))
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Output already exists; choose a new output name.")
    seed_everything(args.seed)
    model, metadata = load_checkpoint(args.checkpoint, resolve_device(args.device))
    images = sample_glyphs(model, args.characters, args.seed, args.variations)
    names = [char for char in args.characters for _ in range(args.variations)]
    save_grid(images, args.output, names, columns=16)
    write_json(
        args.output.with_suffix(".json"),
        {
            "seed": args.seed,
            "characters": args.characters,
            "variations": args.variations,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "quality_status": metadata["quality_status"],
        },
    )
    print(f"Saved {args.output}; checkpoint status: {metadata['quality_status']}")


if __name__ == "__main__":
    main()
