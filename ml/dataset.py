"""Official EMNIST ByClass loading and deterministic train/validation manifests."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Subset
from torchvision.datasets import EMNIST
from torchvision.transforms.functional import pil_to_tensor

from ml.characters import ALPHABET, CONTRACT_VERSION, PREPROCESSING_VERSION
from ml.utils import sha256_file, write_json
from ml.visualize import save_grid


def preprocess(image: Image.Image) -> torch.Tensor:
    # Raw EMNIST is transposed relative to upright display. One transpose restores it.
    upright = image.transpose(Image.Transpose.TRANSPOSE)
    return pil_to_tensor(upright).float().div(255)


def load_emnist(root: str | Path, *, train: bool = True, download: bool = False) -> EMNIST:
    dataset = EMNIST(
        str(root), split="byclass", train=train, download=download, transform=preprocess
    )
    if list(dataset.classes) != list(ALPHABET):
        raise ValueError("Unexpected EMNIST class mapping; do not train until it is inspected.")
    return dataset


def stratified_split(labels, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one.")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError(f"Class {label} needs at least two examples for disjoint partitions.")
        rng.shuffle(indices)
        count = max(1, min(len(indices) - 1, round(len(indices) * fraction)))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    return np.sort(train).astype(np.int64), np.sort(validation).astype(np.int64)


def load_partitions(root: Path, prepared: Path):
    import json

    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["preprocessing_version"] != PREPROCESSING_VERSION
        or manifest["alphabet"] != ALPHABET
        or manifest["contract_version"] != CONTRACT_VERSION
    ):
        raise ValueError("Prepared data uses an incompatible character/preprocessing contract.")
    split_path = prepared / "split.npz"
    if sha256_file(split_path) != manifest["split_sha256"]:
        raise ValueError("Split checksum changed. Prepare data again in a new directory.")
    for filename, expected_hash in manifest["source_file_sha256"].items():
        if "-train-" in filename:
            source = root / "EMNIST" / "raw" / filename
            if not source.is_file() or sha256_file(source) != expected_hash:
                raise ValueError(f"Prepared source data changed: {filename}")
    dataset = load_emnist(root)
    if len(dataset) != manifest["source_train_count"]:
        raise ValueError("Dataset size no longer matches the prepared manifest.")
    with np.load(split_path, allow_pickle=False) as split:
        train, validation = split["train"], split["validation"]
    all_indices = np.concatenate([train, validation])
    if (
        len(all_indices) != len(dataset)
        or len(np.unique(all_indices)) != len(dataset)
        or all_indices.min() != 0
        or all_indices.max() != len(dataset) - 1
    ):
        raise ValueError("Partitions are not disjoint and exhaustive.")
    return dataset, Subset(dataset, train.tolist()), Subset(dataset, validation.tolist()), manifest


def prepare(root: Path, output: Path, *, seed=42, fraction=0.1, download=False):
    torch.set_num_threads(4)
    if (output / "manifest.json").exists() or (output / "split.npz").exists():
        raise FileExistsError(
            f"Prepared data already exists at {output}; choose a new output directory."
        )
    dataset = load_emnist(root, download=download)
    train, validation = stratified_split(dataset.targets.numpy(), fraction, seed)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "split.npz", train=train, validation=validation)
    raw = Path(root) / "EMNIST" / "raw"
    source_files = {
        path.name: sha256_file(path) for path in sorted(raw.glob("emnist-byclass-*-idx*-ubyte"))
    }
    counts = {char: int((dataset.targets == label).sum()) for label, char in enumerate(ALPHABET)}
    manifest = {
        "dataset": "EMNIST",
        "split": "byclass",
        "source": "https://www.nist.gov/itl/products-and-services/emnist-dataset",
        "download_url": EMNIST.url,
        "alphabet": ALPHABET,
        "contract_version": CONTRACT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "seed": seed,
        "validation_fraction": fraction,
        "source_train_count": len(dataset),
        "train_count": len(train),
        "validation_count": len(validation),
        "test_policy": "Official test partition is reserved and was not opened by preparation.",
        "source_file_sha256": source_files,
        "split_sha256": sha256_file(output / "split.npz"),
        "class_counts": counts,
        "class_to_idx": dataset.class_to_idx,
    }
    write_json(output / "manifest.json", manifest)
    examples, names = [], []
    targets = dataset.targets.numpy()
    for label, char in enumerate(ALPHABET):
        indices = np.flatnonzero(targets == label)[:4]
        for index in indices:
            examples.append(dataset[int(index)][0])
            names.append(char)
    save_grid(torch.stack(examples), output / "orientation-and-labels.png", names, columns=16)
    print(f"Prepared {len(train):,} training / {len(validation):,} validation images at {output}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data"))
    parser.add_argument(
        "--download", action="store_true", help="Download the official archive (~536 MiB)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    prepare(
        args.root,
        args.output,
        seed=args.seed,
        fraction=args.validation_fraction,
        download=args.download,
    )


if __name__ == "__main__":
    main()
