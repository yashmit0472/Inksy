import pytest
import torch
from torch.nn import functional as F

from ml.quality import glyph_statistics


def test_blank_and_solid_glyph_statistics_have_known_limits():
    blank = glyph_statistics(torch.zeros(2, 1, 28, 28))
    assert blank["near_blank_fraction"] == 1
    assert blank["mean_edge_energy"] == blank["soft_ink_fraction"] == 0
    solid = glyph_statistics(torch.ones(2, 1, 28, 28))
    assert solid["near_blank_fraction"] == 0
    assert solid["mean_edge_energy"] == solid["soft_ink_fraction"] == 0
    assert solid["mean_ink_intensity"] == 1


def test_edge_diagnostics_detect_smoothing_on_the_same_shape():
    crisp = torch.zeros(1, 1, 28, 28)
    crisp[:, :, 5:23, 12:16] = 1
    smoothed = F.avg_pool2d(crisp, 3, stride=1, padding=1)
    sharp_metrics, soft_metrics = glyph_statistics(crisp), glyph_statistics(smoothed)
    assert soft_metrics["mean_edge_energy"] < sharp_metrics["mean_edge_energy"]
    assert soft_metrics["soft_ink_fraction"] > sharp_metrics["soft_ink_fraction"]


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros(0, 1, 28, 28),
        torch.zeros(1, 28, 28),
        torch.full((1, 1, 28, 28), float("nan")),
        torch.full((1, 1, 28, 28), 1.1),
    ],
)
def test_invalid_images_do_not_produce_diagnostics(images):
    with pytest.raises(ValueError):
        glyph_statistics(images)
