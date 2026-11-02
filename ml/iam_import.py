"""Safely import local IAM archives into a new ignored data directory."""

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from tqdm import tqdm

from ml.utils import sha256_file, write_json


def safe_relative(name: str) -> Path:
    if "\\" in name or ":" in name or "\x00" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    parts = PurePosixPath(name).parts
    if not parts or PurePosixPath(name).is_absolute() or ".." in parts:
        raise ValueError(f"Unsafe archive path: {name!r}")
    if any(part.endswith((".", " ")) for part in parts):
        raise ValueError(f"Unsafe Windows archive path: {name!r}")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
    }
    if any(
        part.split(".", 1)[0].upper() in reserved or any(char in part for char in '<>"|?*')
        for part in parts
    ):
        raise ValueError(f"Unsafe Windows archive path: {name!r}")
    return Path(*parts)


def extract_tar(archive: Path, destination: Path, *, max_bytes=3_000_000_000):
    """Copy regular files only; never ask tarfile to create paths or links."""
    if destination.exists():
        raise FileExistsError(f"Import destination exists: {destination}")
    total = 0
    files = 0
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:*") as source:
        with tqdm(desc=archive.name, unit="file", ascii=True, dynamic_ncols=True) as progress:
            for member in source:
                if member.name in {".", "./"} and member.isdir():
                    continue
                relative = safe_relative(member.name)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(
                        "IAM archives must not contain symbolic/hard links or devices."
                    )
                total += member.size
                if member.size < 0 or member.size > 32_000_000 or total > max_bytes:
                    raise ValueError("IAM archive exceeds the import size limits.")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output)
                files += 1
                progress.update(1)
                if files > 150_000:
                    raise ValueError("IAM archive exceeds the file-count limit.")
    return {"files": files, "unpacked_bytes": total}


def extract_splits(archive: Path, destination: Path):
    if destination.exists():
        raise FileExistsError(f"Import destination exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    total = 0
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            relative = safe_relative(info.filename)
            if info.is_dir():
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Split archives must not contain symbolic links.")
            total += info.file_size
            if total > 5_000_000 or len(source.infolist()) > 100:
                raise ValueError("IAM split archive exceeds the size/count limit.")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output)


def import_archives(downloads: Path, root: Path):
    if root.exists():
        raise FileExistsError("Choose a new IAM data root; existing data is never overwritten.")
    paths = {
        "ascii": downloads / "ascii.tar",
        "words": downloads / "words.tar",
        "splits": downloads / "largeWriterIndependentTextLineRecognitionTask.zip",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing IAM archive: {path}")
    root.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for name in ("ascii", "words"):
        inventory[name] = extract_tar(paths[name], root / name)
    extract_splits(paths["splits"], root / "splits")
    write_json(
        root / "import-manifest.json",
        {
            "dataset": "IAM Handwriting Database 3.0",
            "source": "https://zenodo.org/records/19003721",
            "archive_sha256": {name: sha256_file(path) for name, path in paths.items()},
            "inventory": inventory,
            "training_performed": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path("data/iam-downloads"))
    parser.add_argument("--root", type=Path, default=Path("data/iam"))
    args = parser.parse_args()
    import_archives(args.downloads, args.root)


if __name__ == "__main__":
    main()
