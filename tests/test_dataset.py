import numpy as np
import pytest
import torch
from PIL import Image

from ml.dataset import preprocess, stratified_split


def test_orientation_and_foreground_have_known_coordinates():
    raw = np.zeros((28, 28), dtype=np.uint8)
    raw[3, 17] = 255
    raw[8, 2] = 128
    processed = preprocess(Image.fromarray(raw))
    assert processed.shape == (1, 28, 28)
    assert processed.dtype == torch.float32
    assert processed[0, 17, 3] == 1
    assert processed[0, 2, 8].item() == pytest.approx(128 / 255)
    assert processed[0, 3, 17] == 0


def test_split_is_stratified_disjoint_exhaustive_and_reproducible():
    labels = np.repeat(np.arange(4), [20, 30, 40, 50])
    train, val = stratified_split(labels, 0.2, 42)
    train_again, val_again = stratified_split(labels, 0.2, 42)
    assert np.array_equal(train, train_again)
    assert np.array_equal(val, val_again)
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(range(len(labels)))
    assert np.bincount(labels[val]).tolist() == [4, 6, 8, 10]
    assert not np.array_equal(val, stratified_split(labels, 0.2, 43)[1])


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_invalid_fraction_fails(fraction):
    with pytest.raises(ValueError):
        stratified_split([0, 0, 1, 1], fraction, 1)


def test_insufficient_class_support_fails():
    with pytest.raises(ValueError, match="two examples"):
        stratified_split([0, 1, 1], 0.2, 1)
