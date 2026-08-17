"""Splitting cycles into train / validation / test without leaking between them."""

from typing import NamedTuple

import numpy as np

from shopfloor.data import HEALTHY, PROFILE_COLUMNS, Component

COMPONENTS: tuple[Component, ...] = tuple(HEALTHY)


class Split(NamedTuple):
    """Boolean masks over cycles, one per part."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def run_ids(labels: np.ndarray) -> np.ndarray:
    """Number each contiguous block of cycles that share one component configuration.

    The rig held a configuration fixed for a stretch of cycles — ten on average, up to
    210 — so neighbouring cycles inside a block are near-duplicates: same fault, same
    operating point, minutes apart. A block is therefore the smallest unit that can be
    assigned to a part without the same information appearing on both sides.
    """
    config = labels[:, : len(COMPONENTS)]
    changed = np.any(config[1:] != config[:-1], axis=1)
    return np.concatenate([[0], np.cumsum(changed)])


def levels(labels: np.ndarray) -> dict[Component, list[int]]:
    """The severity grades that occur for each component."""
    return {
        component: sorted(set(labels[:, PROFILE_COLUMNS.index(component)].tolist()))
        for component in COMPONENTS
    }


def missing_levels(labels: np.ndarray, split: Split) -> dict[str, list[int]]:
    """Grades absent from a part. An empty result means the split is usable."""
    expected = levels(labels)
    gaps: dict[str, list[int]] = {}

    for part, mask in zip(split._fields, split, strict=True):
        for component in COMPONENTS:
            column = labels[mask, PROFILE_COLUMNS.index(component)]
            absent = sorted(set(expected[component]) - set(column.tolist()))
            if absent:
                gaps[f"{part}.{component}"] = absent
    return gaps


def split_by_run(
    labels: np.ndarray,
    seed: int,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Split:
    """Assign whole runs to train / val / test, and refuse a split that hides a grade.

    Exact proportions are impossible: one run is 9.5% of all cycles, so the parts land
    near the requested sizes rather than on them.

    Raises if any severity grade is missing from any part — with seed 42 the valve's
    "close to total failure" grade never reaches the test set, which would look like a
    working experiment while quietly never testing the case that matters most.
    """
    runs = run_ids(labels)
    sizes = np.bincount(runs)

    order = np.random.default_rng(seed).permutation(len(sizes))
    reached = np.cumsum(sizes[order]) / len(labels)
    train_end = int(np.searchsorted(reached, 1.0 - val_size - test_size)) + 1
    val_end = int(np.searchsorted(reached, 1.0 - test_size)) + 1

    split = Split(
        *(
            np.isin(runs, group)
            for group in (order[:train_end], order[train_end:val_end], order[val_end:])
        )
    )

    gaps = missing_levels(labels, split)
    if gaps:
        raise ValueError(f"seed {seed} leaves grades unrepresented: {gaps}")
    return split


def find_seed(labels: np.ndarray, start: int = 0, attempts: int = 200, **kwargs: float) -> int:
    """The first seed at or after `start` whose split keeps every grade in every part."""
    for seed in range(start, start + attempts):
        try:
            split_by_run(labels, seed, **kwargs)  # type: ignore[arg-type]
        except ValueError:
            continue
        return seed
    raise ValueError(f"no usable seed in [{start}, {start + attempts})")


if __name__ == "__main__":
    from shopfloor.config import settings

    labels = np.load(settings.processed_dir / "labels.npy")
    split = split_by_run(labels, settings.split_seed)

    print(f"runs: {run_ids(labels).max() + 1}")
    for part, mask in zip(split._fields, split, strict=True):
        print(f"{part:<6} {mask.sum():>5} cycles  {mask.sum() / len(labels):>6.1%}")
    for component, grades in levels(labels).items():
        print(f"{component:<12} {grades}")
