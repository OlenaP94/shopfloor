"""Reading the UCI hydraulic test rig dataset (UCI ML Repository, id 447)."""

from pathlib import Path

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

PROFILE_COLUMNS = ("cooler", "valve", "pump_leak", "accumulator", "stable")

# healthy value for each component
HEALTHY = {"cooler": 100, "valve": 100, "pump_leak": 0, "accumulator": 130}


def read_sensor(path: str | Path) -> list[list[float]]:
    """Read one tab-delimited sensor matrix: rows are cycles, columns are timepoints."""
    with Path(path).open() as f:
        return [[float(v) for v in line.split()] for line in f]


def read_profile(path: str | Path) -> list[dict[str, int]]:
    """Read profile.txt into one dict of component states per cycle."""
    with Path(path).open() as f:
        return [
            dict(zip(PROFILE_COLUMNS, (int(v) for v in line.split()), strict=True)) for line in f
        ]
