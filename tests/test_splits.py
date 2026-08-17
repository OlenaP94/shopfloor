"""Tests for run-aware splitting."""

from itertools import product

import numpy as np
import pytest

from shopfloor.splits import (
    Split,
    find_seed,
    levels,
    missing_levels,
    run_ids,
    split_by_run,
)

GRADES = {
    "cooler": (3, 20, 100),
    "valve": (73, 80, 90, 100),
    "pump_leak": (0, 1, 2),
    "accumulator": (90, 100, 115, 130),
}
SMALL = [(100, 100, 0, 130), (20, 90, 1, 115), (3, 80, 2, 100), (100, 73, 0, 90)]


def build_labels(configurations: list[tuple[int, ...]], per_run: int = 2) -> np.ndarray:
    """A labels matrix of contiguous runs, shaped like profile.txt."""
    rows = [[*config, 0] for config in configurations for _ in range(per_run)]
    return np.array(rows, dtype=np.int32)


@pytest.fixture
def factorial_labels() -> np.ndarray:
    """Every combination of every grade, once — the shape of the real experiment.

    A part must contain all four valve grades, and one run carries exactly one grade,
    so the smallest part needs at least four runs. A handful of configurations can never
    satisfy that; the full 3 x 4 x 3 x 4 factorial gives 144 runs and comfortably does.
    """
    return build_labels(list(product(*GRADES.values())))


def test_run_ids_number_contiguous_blocks() -> None:
    labels = build_labels(SMALL[:2], per_run=3)
    assert run_ids(labels).tolist() == [0, 0, 0, 1, 1, 1]


def test_a_repeated_configuration_starts_a_new_run() -> None:
    """The rig returning to an earlier configuration later is a separate block in time."""
    labels = build_labels([SMALL[0], SMALL[1], SMALL[0]], per_run=2)
    assert run_ids(labels).tolist() == [0, 0, 1, 1, 2, 2]


def test_parts_are_disjoint_and_cover_every_cycle(factorial_labels: np.ndarray) -> None:
    split = split_by_run(factorial_labels, seed=find_seed(factorial_labels))

    stacked = np.vstack(split)
    assert stacked.sum(axis=0).tolist() == [1] * len(factorial_labels)


def test_no_run_is_split_across_parts(factorial_labels: np.ndarray) -> None:
    """The whole point: cycles minutes apart must not land on both sides of the split."""
    split = split_by_run(factorial_labels, seed=find_seed(factorial_labels))
    runs = run_ids(factorial_labels)

    for run in range(runs.max() + 1):
        in_run = runs == run
        assert sum(bool((mask & in_run).any()) for mask in split) == 1


def test_a_usable_split_hides_no_grade(factorial_labels: np.ndarray) -> None:
    split = split_by_run(factorial_labels, seed=find_seed(factorial_labels))
    assert missing_levels(factorial_labels, split) == {}


def test_requested_sizes_are_approached_not_hit(factorial_labels: np.ndarray) -> None:
    """Whole runs cannot land on exact proportions, so the parts only come close."""
    split = split_by_run(factorial_labels, seed=find_seed(factorial_labels))
    shares = [mask.sum() / len(factorial_labels) for mask in split]

    assert shares[0] == pytest.approx(0.70, abs=0.05)
    assert shares[1] == pytest.approx(0.15, abs=0.05)
    assert shares[2] == pytest.approx(0.15, abs=0.05)


def test_missing_levels_names_the_part_and_the_grades(factorial_labels: np.ndarray) -> None:
    everything = np.ones(len(factorial_labels), dtype=bool)
    nothing = np.zeros(len(factorial_labels), dtype=bool)
    split = Split(train=everything, val=nothing, test=nothing)

    gaps = missing_levels(factorial_labels, split)
    assert gaps["val.cooler"] == [3, 20, 100]
    assert "train.cooler" not in gaps


def test_split_by_run_refuses_a_seed_that_hides_a_grade() -> None:
    """Four runs cannot put all four valve grades in all three parts, whatever the seed."""
    labels = build_labels(SMALL, per_run=8)

    with pytest.raises(ValueError, match="unrepresented"):
        for seed in range(50):
            split_by_run(labels, seed)


def test_find_seed_gives_up_rather_than_looping_forever() -> None:
    labels = build_labels(SMALL[:2], per_run=4)

    with pytest.raises(ValueError, match="no usable seed"):
        find_seed(labels, attempts=10)


def test_levels_lists_the_grades_present(factorial_labels: np.ndarray) -> None:
    assert levels(factorial_labels) == {name: list(grades) for name, grades in GRADES.items()}
