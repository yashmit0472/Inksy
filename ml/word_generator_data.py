"""Deterministic single-writer scope; original IAM partitions remain intact."""

import hashlib
import re
from pathlib import Path

from ml.iam import IAMWordDataset, collate_words
from ml.utils import sha256_file


def load_scope(prepared, writer_id="000", seed=42, expected_records=None):
    prepared = Path(prepared)
    data = IAMWordDataset(prepared, "train")
    digest = data.manifest["records_sha256"]
    if expected_records is not None and expected_records != digest:
        raise ValueError("Prepared data differs from the configured record digest.")
    eligible = [
        index
        for index, row in enumerate(data.rows)
        if row["writer_id"] == writer_id and re.fullmatch(r"[a-z]{2,10}", row["text"])
    ]
    ordered = sorted(
        eligible,
        key=lambda index: hashlib.sha256(
            f"inksy-generation-diagnostic:{seed}:{data.rows[index]['id']}".encode()
        ).hexdigest(),
    )
    selected, words = [], set()
    for index in ordered:
        text = data.rows[index]["text"]
        if text not in words:
            selected.append(index)
            words.add(text)
        if len(selected) == 32:
            break
    if len(selected) != 32:
        raise ValueError("This writer needs at least 32 distinct eligible training words.")
    identity = {
        "writer_id": writer_id,
        "partition": "train",
        "seed": seed,
        "records_sha256": digest,
        "manifest_sha256": sha256_file(prepared / "manifest.json"),
        "scope": "original lowercase ASCII labels, lengths 2–10",
        "eligible_ids": [data.rows[index]["id"] for index in eligible],
        "diagnostic_ids": [data.rows[index]["id"] for index in selected],
        "diagnostic_texts": [data.rows[index]["text"] for index in selected],
        "selection": "First instance per word in seeded SHA256 order; training only.",
    }
    return data, selected, eligible, identity


def image_batch(items):
    batch = collate_words(items)
    return batch["images"], batch["widths"], batch["text"]
