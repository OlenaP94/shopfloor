"""Tests for the per-window feature table."""

from pathlib import Path

import numpy as np
import pytest

from shopfloor.arrays import build_tensor
from shopfloor.features import (
    N_WINDOWS,
    STATISTICS,
    VIRTUAL,
    feature_names,
    virtual_mask,
    window_features,
)


def column(channel: int, window: int, statistic: str) -> int:
    """Where one statistic of one window of one channel lands in the flat table."""
    return (channel * N_WINDOWS + window) * len(STATISTICS) + STATISTICS.index(statistic)


def test_every_channel_window_and_statistic_gets_a_column(tiny_dataset: Path) -> None:
    tensor, channels = build_tensor(tiny_dataset)
    features = window_features(tensor)

    assert features.shape == (len(tensor), len(channels) * N_WINDOWS * len(STATISTICS))
    assert features.dtype == np.float32
    assert features.shape[1] == len(feature_names(channels))


def test_names_run_channel_then_window_then_statistic(tiny_dataset: Path) -> None:
    _, channels = build_tensor(tiny_dataset)
    names = feature_names(channels)

    assert names[0] == "PS1_mean_w0_mean"
    assert names[5] == "PS1_mean_w0_p75"
    assert names[6] == "PS1_mean_w1_mean"
    assert names[-1] == f"SE_w{N_WINDOWS - 1}_p75"


def test_columns_land_where_the_names_say_they_do(tiny_dataset: Path) -> None:
    """The one test that keeps window_features and feature_names from drifting apart."""
    tensor, channels = build_tensor(tiny_dataset)
    features = window_features(tensor)
    width = tensor.shape[2] // N_WINDOWS
    index = channels.index("PS1_mean")

    for window in (0, N_WINDOWS - 1):
        chunk = tensor[:, index, window * width : (window + 1) * width]
        assert np.allclose(features[:, column(index, window, "mean")], chunk.mean(axis=1))
        assert np.allclose(features[:, column(index, window, "max")], chunk.max(axis=1))


def test_virtual_mask_covers_exactly_the_computed_channels(tiny_dataset: Path) -> None:
    _, channels = build_tensor(tiny_dataset)
    names = feature_names(channels)
    mask = virtual_mask(channels)

    assert mask.sum() == len(VIRTUAL) * N_WINDOWS * len(STATISTICS)
    flagged = [name for name, is_virtual in zip(names, mask, strict=True) if is_virtual]
    assert all(name.split("_")[0] in VIRTUAL for name in flagged)


def test_windows_must_divide_the_cycle_evenly() -> None:
    with pytest.raises(ValueError, match="do not divide"):
        window_features(np.zeros((2, 3, 100), dtype=np.float32), n_windows=7)
