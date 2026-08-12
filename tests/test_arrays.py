"""Tests for resampling the raw sensor files into one tensor."""

from pathlib import Path

import numpy as np
import pytest

from shopfloor.arrays import (
    TARGET_POINTS,
    build_tensor,
    check_matrix,
    read_labels,
    read_matrix,
)
from shopfloor.data import PROFILE_COLUMNS
from shopfloor.dataset import HydraulicDataError


def test_tensor_has_the_expected_shape_and_dtype(tiny_dataset: Path) -> None:
    tensor, names = build_tensor(tiny_dataset)
    n_cycles = len(read_labels(tiny_dataset))
    assert tensor.shape == (n_cycles, 24, TARGET_POINTS)
    assert tensor.dtype == np.float32
    assert len(names) == 24


def test_fast_sensors_yield_a_mean_and_a_std_channel(tiny_dataset: Path) -> None:
    _, names = build_tensor(tiny_dataset)
    assert names[:4] == ["PS1_mean", "PS1_std", "PS2_mean", "PS2_std"]
    assert names[-3:] == ["CE", "CP", "SE"]
    assert sum(1 for name in names if name.endswith("_std")) == 7


def test_fast_sensor_is_block_averaged(tiny_dataset: Path) -> None:
    """The 100 Hz PS1 is cut into 600 blocks of 10 samples, not decimated."""
    tensor, names = build_tensor(tiny_dataset)
    raw = read_matrix(tiny_dataset / "PS1.txt")

    means = tensor[0, names.index("PS1_mean")]
    assert np.isclose(means[0], raw[0, :10].mean())
    assert np.isclose(means[1], raw[0, 10:20].mean())
    assert np.isclose(means[-1], raw[0, -10:].mean())

    stds = tensor[0, names.index("PS1_std")]
    assert np.isclose(stds[0], raw[0, :10].std())


def test_sensor_already_at_target_rate_is_untouched(tiny_dataset: Path) -> None:
    tensor, names = build_tensor(tiny_dataset)
    raw = read_matrix(tiny_dataset / "FS1.txt")
    assert np.array_equal(tensor[0, names.index("FS1")], raw[0])


def test_slow_sensor_is_repeated_not_interpolated(tiny_dataset: Path) -> None:
    """Every 1 Hz reading is held for ten timepoints, so no new values appear."""
    tensor, names = build_tensor(tiny_dataset)
    raw = read_matrix(tiny_dataset / "TS1.txt")
    channel = tensor[0, names.index("TS1")]

    assert np.array_equal(channel[:10], np.full(10, raw[0, 0]))
    assert np.array_equal(channel[::10], raw[0])


def test_labels_are_integers_one_column_per_profile_field(tiny_dataset: Path) -> None:
    labels = read_labels(tiny_dataset)
    assert labels.dtype == np.int32
    assert labels.shape[1] == len(PROFILE_COLUMNS)


def test_check_matrix_rejects_the_wrong_number_of_timepoints() -> None:
    with pytest.raises(HydraulicDataError, match="points per cycle"):
        check_matrix("PS1", np.zeros((3, 5000), dtype=np.float32), 100)


def test_check_matrix_rejects_a_one_dimensional_file() -> None:
    with pytest.raises(HydraulicDataError, match="2-D"):
        check_matrix("PS1", np.zeros(6000, dtype=np.float32), 100)


def test_sensors_must_agree_on_the_cycle_count(tiny_dataset: Path) -> None:
    path = tiny_dataset / "TS1.txt"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")

    with pytest.raises(HydraulicDataError, match="cycle count"):
        build_tensor(tiny_dataset)
