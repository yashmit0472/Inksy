import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-training",
        action="store_true",
        default=False,
        help="Opt in to synthetic tests that update model weights. Run manually only.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-training"):
        return
    skip = pytest.mark.skip(reason="Weight-update tests require an explicit --run-training opt-in.")
    for item in items:
        if "training" in item.keywords:
            item.add_marker(skip)


def pytest_sessionstart(session):
    import torch

    torch.set_num_threads(2)
