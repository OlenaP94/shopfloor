"""Random forest baselines: the floor every later model has to beat.

Scored on validation, never on test. The test set is spent the moment it is looked at
more than once — every glance leaks a little information into the next decision — so it
stays sealed until the final comparison.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from shopfloor.config import settings
from shopfloor.data import HEALTHY, PROFILE_COLUMNS
from shopfloor.features import virtual_mask
from shopfloor.metrics import (
    accuracy,
    alarm_rates,
    confusion_matrix,
    macro_f1,
    matrix_report,
    report,
)
from shopfloor.splits import COMPONENTS, levels, split_by_run

N_TREES = 300


def normalise(features: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Centre and scale every column using the training rows only.

    Statistics taken from the whole array would let validation and test influence the
    numbers that rescale them. The metrics improve and nothing warns you.

    Constant columns keep a divisor of 1: they carry no information either way, and
    dividing by their zero spread would poison the array with infinities.
    """
    mean = features[train].mean(axis=0)
    spread = features[train].std(axis=0)
    return ((features - mean) / np.where(spread == 0, 1.0, spread)).astype(np.float32)


def fit(
    features: np.ndarray, target: np.ndarray, train: np.ndarray, seed: int
) -> RandomForestClassifier:
    """A deliberately plain forest — a baseline is a floor, not an attempt to win."""
    forest = RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=-1)
    forest.fit(features[train], target[train])
    return forest


def score(
    target: np.ndarray,
    predicted: np.ndarray,
    grades: list[int],
    healthy: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Grading metrics, operational alarm rates, and the matrix they were derived from.

    Takes predictions rather than a model: scoring has no business knowing that a random
    forest produced these numbers, and the same function will score the network in week 4.
    """
    matrix = confusion_matrix(target, predicted, grades)
    false_alarm, missed_alarm = alarm_rates(target, predicted, healthy)
    return {
        "macro_f1": macro_f1(matrix),
        "accuracy": accuracy(matrix),
        "far": false_alarm,
        "mar": missed_alarm,
    }, matrix


def top_features(forest: RandomForestClassifier, names: list[str], count: int = 5) -> list[str]:
    """The columns the forest leaned on most, to compare against what the EDA predicted."""
    order = np.argsort(forest.feature_importances_)[::-1][:count]
    return [names[i] for i in order]


if __name__ == "__main__":
    root = settings.processed_dir
    features = np.load(root / "features.npy")
    labels = np.load(root / "labels.npy")
    names = (root / "feature_names.txt").read_text().split()
    channels = (root / "channels.txt").read_text().split()

    split = split_by_run(labels, settings.split_seed)
    scaled = normalise(features, split.train)
    grades = levels(labels)

    every = np.ones(len(names), dtype=bool)
    variants: dict[str, np.ndarray] = {"all": every, "measured": ~virtual_mask(channels)}

    header = (
        f"{'component':<12} {'channels':<9} {'macro F1':>9} {'accuracy':>9} {'FAR':>6} {'MAR':>6}"
    )
    lines = [header, "-" * len(header)]
    matrices: list[str] = []

    for component in COMPONENTS:
        target = labels[:, PROFILE_COLUMNS.index(component)]

        for variant, columns in variants.items():
            subset = scaled[:, columns]
            forest = fit(subset, target, split.train, settings.seed)
            predicted = forest.predict(subset[split.val])
            result, matrix = score(
                target[split.val], predicted, grades[component], HEALTHY[component]
            )
            lines.append(
                f"{component:<12} {variant:<9} {result['macro_f1']:>9.3f} "
                f"{result['accuracy']:>9.3f} {result['far']:>6.3f} {result['mar']:>6.3f}"
            )

            if variant == "measured":
                kept = [name for name, keep in zip(names, columns, strict=True) if keep]
                lines.append(f"{'':<12} relies on: {', '.join(top_features(forest, kept))}")
                matrices.append(
                    f"\n{component} — measured channels only\n"
                    f"{report(matrix, grades[component])}\n\n"
                    f"{matrix_report(matrix, grades[component])}"
                )

    summary = "\n".join(lines + matrices)
    print(summary)

    destination = settings.reports_dir / "baseline_val.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(summary + "\n")
    print(f"\nwritten to {destination}")
