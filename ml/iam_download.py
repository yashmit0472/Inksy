"""Download the author-authorized IAM migration; no credentials or model training."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from tqdm import tqdm

from ml.utils import sha256_file, write_json

RECORD_URL = "https://zenodo.org/records/19003721"
API_URL = "https://zenodo.org/api/records/19003721"
# Checksums published by the dataset's author-authorized migration, checked 29 Oct 2026.
FILES = {
    "ascii.tar": "67eb823a5204e5924481e15089adbbdd",
    "words.tar": "4a71f8709d89733580bffbe83e072a29",
    "largeWriterIndependentTextLineRecognitionTask.zip": "4803883ed7f52be3c9fa759f538baa94",
}


def md5_file(path):
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API_URL, timeout=60) as response:
        metadata = json.load(response)
    entries = {entry["key"]: entry for entry in metadata["files"]}
    for name, checksum in FILES.items():
        if entries[name]["checksum"] != f"md5:{checksum}":
            raise ValueError(
                "Published IAM checksum changed; inspect provenance before proceeding."
            )
    write_json(output / "zenodo-record.json", metadata)
    records = {}
    for name, expected in FILES.items():
        path = output / name
        url = f"{RECORD_URL}/files/{name}?download=1"
        if path.exists():
            if md5_file(path) != expected:
                raise ValueError(f"Existing {path} has the wrong checksum; it was not overwritten.")
            print(f"Verified existing {path}", flush=True)
        else:
            temporary = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(url, timeout=60) as response:
                with (
                    temporary.open("wb") as stream,
                    tqdm(
                        total=entries[name]["size"],
                        desc=name,
                        unit="B",
                        unit_scale=True,
                        ascii=True,
                        dynamic_ncols=True,
                    ) as progress,
                ):
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
                        progress.update(len(block))
            if md5_file(temporary) != expected:
                raise ValueError(f"IAM download checksum mismatch: {name}; partial file retained.")
            temporary.replace(path)
        records[name] = {
            "url": url,
            "bytes": path.stat().st_size,
            "md5": expected,
            "sha256": sha256_file(path),
        }
        write_json(
            output / "download-manifest.json",
            {
                "dataset": "IAM Handwriting Database 3.0",
                "source": RECORD_URL,
                "terms": "Non-commercial research purposes; cite Marti and Bunke (2002).",
                "files": records,
                "training_performed": False,
            },
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/iam-downloads"))
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
