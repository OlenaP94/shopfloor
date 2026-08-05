"""Reading and summarising SKAB sensor data."""

import csv
import statistics
from pathlib import Path

SEPARATOR = ";"
NON_NUMERIC = {"datetime"}
LABELS = {"anomaly", "changepoint"}


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a SKAB CSV file and return a list of row dicts."""
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter=SEPARATOR))


def column_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """Return {column: {"min", "max", "mean", "std"}} for numeric columns."""
    if not rows:
        return {}

    stats: dict[str, dict[str, float]] = {}
    numeric_cols = [c for c in rows[0] if c not in NON_NUMERIC]

    for col in numeric_cols:
        vals = [float(r[col]) for r in rows]
        stats[col] = {
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }

    return stats


def split_by_label(
    rows: list[dict[str, str]], label: str = "anomaly"
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split rows into (normal, anomalous) by a binary 0/1 label column."""
    normal = [r for r in rows if float(r[label]) == 0]
    anomalous = [r for r in rows if float(r[label]) == 1]
    return normal, anomalous


def compare_groups(
    normal: list[dict[str, str]],
    anomalous: list[dict[str, str]],
    metric: str = "mean",
) -> dict[str, float]:
    """Return {sensor: percent change in the given metric, normal → anomalous}."""
    if not normal or not anomalous:
        return {}

    stats_normal = column_stats(normal)
    stats_anomalous = column_stats(anomalous)
    sensors = [c for c in stats_normal if c not in LABELS]

    change: dict[str, float] = {}
    for col in sensors:
        base = stats_normal[col][metric]
        if base == 0:
            continue
        change[col] = (stats_anomalous[col][metric] - base) / base * 100

    return change


def print_comparison(
    normal: list[dict[str, str]],
    anomalous: list[dict[str, str]],
    metric: str,
) -> None:
    """Print a sorted comparison table for one metric."""
    change = compare_groups(normal, anomalous, metric)
    stats_normal = column_stats(normal)
    stats_anomalous = column_stats(anomalous)

    print(f"\n=== {metric.upper()} ===")
    print(f"{'sensor':<22}{'normal':>12}{'anomalous':>12}{'change':>10}")
    for col, pct in sorted(change.items(), key=lambda kv: abs(kv[1]), reverse=True):
        print(
            f"{col:<22}"
            f"{stats_normal[col][metric]:>12.4f}"
            f"{stats_anomalous[col][metric]:>12.4f}"
            f"{pct:>9.1f}%"
        )


if __name__ == "__main__":
    rows = read_csv("data/raw/valve1_0.csv")
    normal, anomalous = split_by_label(rows)

    print(f"Rows: {len(rows)}  normal: {len(normal)}  anomalous: {len(anomalous)}")
    print(f"Anomalous share: {len(anomalous) / len(rows):.1%}")

    indices = [i for i, r in enumerate(rows) if float(r["anomaly"]) == 1]
    contiguous = bool(indices) and max(indices) - min(indices) + 1 == len(indices)
    if indices:
        print(f"Anomaly rows: {min(indices)}–{max(indices)}, count {len(indices)}")
        print(f"Contiguous: {contiguous}")

    # Naive comparison: whole normal group vs whole anomalous group.
    # Confounded, because the normal group spans a longer and discontinuous
    # time window, so any drifting sensor shows an inflated std there.
    print_comparison(normal, anomalous, "mean")
    print_comparison(normal, anomalous, "std")

    # Controlled comparison: the anomaly window against an equally long window
    # immediately before it. Adjacent in time and the same length, so slow
    # drift affects both roughly equally.
    if contiguous:
        start, end = min(indices), max(indices)
        n = len(indices)
        before = rows[max(0, start - n) : start]
        during = rows[start : end + 1]
        print(f"\n--- controlled: {len(before)} rows before vs {len(during)} during ---")
        print_comparison(before, during, "mean")
        print_comparison(before, during, "std")
