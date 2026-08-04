"""Reading and summarising SKAB sensor data."""

import csv
from pathlib import Path

SEPARATOR = ";"
NON_NUMERIC = {"datetime"}


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a SKAB CSV file and return a list of row dicts."""
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter=SEPARATOR))


def column_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """Return {column: {"min": .., "max": .., "mean": ..}} for numeric columns."""
    stats: dict[str, dict[str, float]] = {}
    numeric_cols = [c for c in rows[0] if c not in NON_NUMERIC]

    for col in numeric_cols:
        vals = [float(r[col]) for r in rows]
        stats[col] = {
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
        }

    return stats


if __name__ == "__main__":
    rows = read_csv("data/raw/valve1_0.csv")
    print(f"Rows: {len(rows)}")
    for col, s in column_stats(rows).items():
        print(f"{col:<22} min={s['min']:>10.4f}  max={s['max']:>10.4f}  mean={s['mean']:>10.4f}")
