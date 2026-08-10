"""Reading the UCI hydraulic test rig dataset (UCI ML Repository, id 447)."""

from pathlib import Path
from typing import Literal

# --- types ----------------------------------------------------------------

type ProfileColumn = Literal["cooler", "valve", "pump_leak", "accumulator", "stable"]
"""Any column of profile.txt, including the stability flag."""

type Component = Literal["cooler", "valve", "pump_leak", "accumulator"]
"""A hydraulic component that has a healthy state. Excludes "stable"."""

type SensorMatrix = list[list[float]]
"""One sensor: rows are cycles, columns are timepoints within a cycle."""

type Profile = list[dict[ProfileColumn, int]]
"""Component states, one dict per cycle."""

# --- constants ------------------------------------------------------------
# PROFILE_COLUMNS must stay in sync with ProfileColumn above. A Literal cannot
# be derived from a tuple: the tuple exists at runtime, the Literal only during
# type checking, so the values are deliberately written twice.

PROFILE_COLUMNS: tuple[ProfileColumn, ...] = (
    "cooler",
    "valve",
    "pump_leak",
    "accumulator",
    "stable",
)

HEALTHY: dict[Component, int] = {
    "cooler": 100,
    "valve": 100,
    "pump_leak": 0,
    "accumulator": 130,
}

# sensor name -> sampling rate in Hz
SENSORS: dict[str, int] = {
    "PS1": 100,
    "PS2": 100,
    "PS3": 100,
    "PS4": 100,
    "PS5": 100,
    "PS6": 100,
    "EPS1": 100,
    "FS1": 10,
    "FS2": 10,
    "TS1": 1,
    "TS2": 1,
    "TS3": 1,
    "TS4": 1,
    "VS1": 1,
    "CE": 1,
    "CP": 1,
    "SE": 1,
}

CYCLE_SECONDS = 60
N_CYCLES = 2205


# --- readers --------------------------------------------------------------


def read_sensor(path: str | Path) -> SensorMatrix:
    """Read one tab-delimited sensor matrix: rows are cycles, columns are timepoints."""
    with Path(path).open() as f:
        return [[float(v) for v in line.split()] for line in f]


def read_profile(path: str | Path) -> Profile:
    """Read profile.txt into one dict of component states per cycle."""
    with Path(path).open() as f:
        return [
            dict(zip(PROFILE_COLUMNS, (int(v) for v in line.split()), strict=True)) for line in f
        ]
