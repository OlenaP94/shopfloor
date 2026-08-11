"""Shared fixtures. pytest finds this file automatically — no import needed."""

from pathlib import Path

import pytest

from shopfloor.data import CYCLE_SECONDS, SENSORS

# Three cycles with known states, so tests can assert exact counts.
#      cooler  valve  pump  accumulator  stable
PROFILE_ROWS = [
    (100, 100, 0, 130, 0),  # healthy in all four
    (3, 73, 2, 90, 1),  # broken in all four
    (100, 90, 1, 115, 0),  # cooler healthy, the rest degraded
]


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Build a miniature but structurally valid dataset. Returns its directory."""
    root = tmp_path / "hydraulic"
    root.mkdir()

    (root / "profile.txt").write_text(
        "\n".join("\t".join(str(v) for v in row) for row in PROFILE_ROWS) + "\n"
    )

    for name, rate in SENSORS.items():
        n_points = rate * CYCLE_SECONDS
        row = "\t".join(str(float(i)) for i in range(n_points))
        (root / f"{name}.txt").write_text("\n".join([row] * len(PROFILE_ROWS)) + "\n")

    return root
