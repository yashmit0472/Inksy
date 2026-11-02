"""Training-only word-frequency weighting; never derive weights from holdouts."""

import hashlib
import json
from collections import Counter

import torch

from ml.iam import collate_words


class FrequencyBalancedCollator:
    """Bound rare/common per-example weight ratio to ten, retaining all examples."""

    def __init__(self, training_rows):
        if not training_rows or any(row["partition"] != "train" for row in training_rows):
            raise ValueError("Frequency weights must use nonempty training records only.")
        self.counts = Counter(row["text"].casefold() for row in training_rows)
        self.weights = {text: max(0.1, count**-0.5) for text, count in self.counts.items()}

    def __call__(self, items):
        texts = [item["text"].casefold() for item in items]
        if any(text not in self.weights for text in texts):
            raise ValueError("Training batch contains a word absent from training frequencies.")
        batch = collate_words(items)
        batch["reader_loss_weights"] = torch.tensor(
            [self.weights[text] for text in texts], dtype=torch.float32
        )
        return batch

    def report(self):
        examples = sum(self.counts.values())
        mass = sum(count * self.weights[text] for text, count in self.counts.items())
        common = self.counts.most_common(10)
        canonical = json.dumps(dict(self.counts), sort_keys=True, ensure_ascii=False).encode()
        return {
            "mode": "inverse-sqrt-word-frequency-clipped",
            "counts_partition": "train",
            "casefold_grouping": True,
            "frequency_sha256": hashlib.sha256(canonical).hexdigest(),
            "examples": examples,
            "distinct_casefold_words": len(self.counts),
            "formula": "max(0.1, training_count(casefold(text)) ** -0.5)",
            "maximum_per_example_weight_ratio": 10,
            "unweighted_top10_example_share": sum(count for _, count in common) / examples,
            "weighted_top10_dataset_mass_share": sum(
                count * self.weights[text] for text, count in common
            )
            / mass,
            "note": "Batch-normalized weights; dataset mass share is not an exact gradient share.",
        }
