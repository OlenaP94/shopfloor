"""A validated wrapper around a single SKAB experiment file."""

from pathlib import Path

from shopfloor.data import LABELS, NON_NUMERIC, read_csv

REQUIRED_COLUMNS = {
    "datetime",
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
    "anomaly",
    "changepoint",
}


class SkabDataError(Exception):
    """Raised when a SKAB file does not have the expected structure."""


class SkabDataset:
    """One SKAB experiment loaded from CSV, with its columns validated."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rows = read_csv(self.path)
        self._validate()

    def _validate(self) -> None:
        """Raise SkabDataError if the file is empty or missing columns."""
        if not self.rows:
            raise SkabDataError(f"{self.path.name} contains no data rows")

        missing = REQUIRED_COLUMNS - set(self.rows[0])
        if missing:
            raise SkabDataError(f"{self.path.name} is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]

    def __repr__(self) -> str:
        return (
            f"SkabDataset({self.path.name!r}, "
            f"{len(self)} rows, {len(self.anomaly_indices)} anomalous)"
        )

    @property
    def sensor_columns(self) -> list[str]:
        """Column names holding sensor readings — no timestamp, no labels."""
        return [c for c in self.rows[0] if c not in NON_NUMERIC and c not in LABELS]

    @property
    def anomaly_indices(self) -> list[int]:
        """Row positions where anomaly == 1."""
        return [i for i, r in enumerate(self.rows) if float(r["anomaly"]) == 1]


if __name__ == "__main__":
    ds = SkabDataset("data/raw/valve1_0.csv")
    print(ds)
    print(f"len:             {len(ds)}")
    print(f"sensor columns:  {ds.sensor_columns}")
    print(f"first row time:  {ds[0]['datetime']}")
    print(f"anomaly span:    {min(ds.anomaly_indices)}–{max(ds.anomaly_indices)}")
    print(f"iterable:        {sum(1 for _ in ds)} rows by iteration")

    broken = Path("data/raw/broken.csv")
    broken.write_text("datetime;Current\n2020-03-09 10:00:00;1.0\n")
    try:
        SkabDataset(broken)
    except SkabDataError as e:
        print(f"\ncaught as expected:\n  {e}")
    finally:
        broken.unlink(missing_ok=True)
