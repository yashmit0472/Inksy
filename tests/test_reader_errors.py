import itertools
import math
from collections import defaultdict

import pytest
import torch

from ml.reader_errors import edit_operations, prefix_beam, summarize_errors
from ml.word_reader import edit_distance


def collapse(path):
    return tuple(
        token for i, token in enumerate(path) if token and (i == 0 or token != path[i - 1])
    )


def test_prefix_beam_matches_exhaustive_ctc_paths_including_repeated_letters():
    probabilities = torch.tensor(
        [[0.2, 0.5, 0.3], [0.5, 0.2, 0.3], [0.2, 0.7, 0.1], [0.3, 0.1, 0.6]], dtype=torch.float64
    )
    summed = defaultdict(float)
    for path in itertools.product(range(3), repeat=4):
        summed[collapse(path)] += math.prod(
            float(probabilities[i, token]) for i, token in enumerate(path)
        )
    expected = max(summed, key=summed.get)
    actual, score = prefix_beam(probabilities.log(), beam_size=100)
    assert actual == expected
    assert math.exp(score) == pytest.approx(summed[expected], abs=1e-12)


@pytest.mark.parametrize(
    "expected,predicted",
    [
        ("book", "bok"),
        ("discuss", "das"),
        ("a", "aaaa"),
        ("", "a"),
        ("cat", ""),
        ("Labour", "dar"),
        ("same", "same"),
    ],
)
def test_error_decomposition_matches_levenshtein(expected, predicted):
    assert len(edit_operations(expected, predicted)) == edit_distance(expected, predicted)


def test_length_buckets_and_insertions_are_reported_without_clipping():
    rows = [{"expected": "a", "predicted": "aaa"}, {"expected": "abcdefgh", "predicted": ""}]
    report = summarize_errors(rows)
    assert report["character_edits"] == 10
    assert report["edit_counts"] == {"insertion": 2, "deletion": 8}
    assert report["by_reference_length"]["1-3"]["character_error_rate"] == 2
    assert report["by_reference_length"]["7+"]["examples"] == 1
