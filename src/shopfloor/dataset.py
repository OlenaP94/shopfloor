"""A validated wrapper around the UCI hydraulic test rig dataset."""

from pathlib import Path
from typing import TypedDict

from shopfloor.config import settings
from shopfloor.data import (
    CYCLE_SECONDS,
    HEALTHY,
    PROFILE_COLUMNS,
    SENSORS,
    Component,
    ProfileColumn,
    read_profile,
    read_sensor,
)


class Cycle(TypedDict):
    """One cycle: the raw signal of every sensor plus the four component states."""

    signals: dict[str, list[float]]
    labels: dict[ProfileColumn, int]


class HydraulicDataError(Exception):
    """Raised when the dataset directory is missing files or has wrong shapes."""


class HydraulicDataset:
    """The hydraulic rig dataset: 17 sensors and 4 component states per cycle."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._check_files()
        self.profile = read_profile(self.root / "profile.txt")
        self.sensors = {name: read_sensor(self.root / f"{name}.txt") for name in SENSORS}
        self._check_shapes()

    def _check_files(self) -> None:
        """Raise if the directory or any expected file is missing."""
        if not self.root.exists():
            raise HydraulicDataError(f"{self.root} does not exist — run `make data` first")

        expected = {f"{name}.txt" for name in SENSORS} | {"profile.txt"}
        missing = expected - {p.name for p in self.root.glob("*.txt")}
        if missing:
            raise HydraulicDataError(f"missing files: {sorted(missing)}")

    def _check_shapes(self) -> None:
        """Raise if the profile is empty or any sensor matrix disagrees with it."""
        n_cycles = len(self.profile)
        if n_cycles == 0:
            raise HydraulicDataError(f"{self.root / 'profile.txt'} has no rows")

        for name, rate in SENSORS.items():
            matrix = self.sensors[name]
            expected_points = rate * CYCLE_SECONDS
            if len(matrix) != n_cycles:
                raise HydraulicDataError(f"{name}: {len(matrix)} cycles, expected {n_cycles}")
            if len(matrix[0]) != expected_points:
                raise HydraulicDataError(
                    f"{name}: {len(matrix[0])} points, expected {expected_points}"
                )

    def __len__(self) -> int:
        return len(self.profile)

    def __getitem__(self, index: int) -> Cycle:
        return {
            "signals": {name: matrix[index] for name, matrix in self.sensors.items()},
            "labels": self.profile[index],
        }

    def __repr__(self) -> str:
        return f"HydraulicDataset({self.root.name!r}, {len(self)} cycles, {len(SENSORS)} sensors)"

    def labels(self, component: ProfileColumn) -> list[int]:
        """All cycle values for one column of profile.txt, e.g. "valve"."""
        if component not in PROFILE_COLUMNS:
            raise HydraulicDataError(f"unknown column {component!r}")
        return [row[component] for row in self.profile]

    def healthy_indices(self, component: Component | None = None) -> list[int]:
        """Cycles healthy in one component, or in all four if component is None."""
        if component is not None:
            if component not in HEALTHY:
                raise HydraulicDataError(f"unknown component {component!r}")
            target = HEALTHY[component]
            return [i for i, row in enumerate(self.profile) if row[component] == target]
        return [
            i for i, row in enumerate(self.profile) if all(row[c] == v for c, v in HEALTHY.items())
        ]


if __name__ == "__main__":
    ds = HydraulicDataset(settings.data_dir)
    print(ds)

    cycle = ds[0]
    print(f"labels[0]:        {cycle['labels']}")
    print(f"PS1 points:       {len(cycle['signals']['PS1'])}")
    print(f"TS1 points:       {len(cycle['signals']['TS1'])}")

    for component in HEALTHY:
        n = len(ds.healthy_indices(component))
        print(f"healthy {component:<12} {n:>5}  ({n / len(ds):.1%})")

    print(f"healthy in all four: {len(ds.healthy_indices())}")
