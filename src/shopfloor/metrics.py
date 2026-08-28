"""Evaluation metrics for severity grading, built from the confusion matrix up."""

from collections.abc import Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Counts of true class (rows) against predicted class (columns), in `labels` order.

    The classes come from `labels`, never from the data: a split that happens to contain
    no "close to total failure" cycles must still produce a matrix of the same shape,
    or metrics stop being comparable between runs.
    """
    position = {label: i for i, label in enumerate(labels)}
    unexpected = (set(np.unique(y_true)) | set(np.unique(y_pred))) - set(labels)
    if unexpected:
        raise ValueError(f"values not in labels: {sorted(unexpected)}")

    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true, predicted in zip(y_true, y_pred, strict=True):
        matrix[position[true], position[predicted]] += 1
    return matrix


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division that yields 0.0 wherever the denominator is 0."""
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)


def precision_recall_f1(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class precision, recall and F1.

    A class that was never predicted and never occurred gives 0/0. That is undefined,
    not zero, but a metric has to return something — we return 0.0, which is the
    convention scikit-learn uses under zero_division=0. Stated here because different
    libraries choose differently and the numbers then disagree.
    """
    true_positive = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)  # column: how often we said this class
    actual = cm.sum(axis=1).astype(np.float64)  # row: how often the class occurred

    precision = _divide(true_positive, predicted)
    recall = _divide(true_positive, actual)
    f1 = _divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def macro_f1(cm: np.ndarray) -> float:
    """Unweighted mean F1 across classes, so a rare severity counts as much as a common one."""
    return float(precision_recall_f1(cm)[2].mean())


def accuracy(cm: np.ndarray) -> float:
    """Share of cycles graded exactly right. Reported for context, never on its own."""
    total = cm.sum()
    return float(np.diag(cm).sum() / total) if total else 0.0


def alarm_rates(y_true: np.ndarray, y_pred: np.ndarray, healthy: int) -> tuple[float, float]:
    """False-alarm and missed-alarm rate, after collapsing severities to healthy / faulty.

    FAR: of the cycles that were healthy, the share flagged as faulty — wasted callouts.
    MAR: of the cycles that were faulty, the share passed as healthy — the expensive kind.

    A fault graded at the wrong severity is not a missed alarm: the alarm still fired.
    That is why these two numbers say something macro F1 does not.

    With no healthy cycles the false-alarm rate does not exist, and likewise for the
    missed-alarm rate with no faulty ones. Those cases return nan rather than 0.0:
    "no false alarms" and "false alarms could not be measured" must not look alike.
    """
    truly_faulty = np.asarray(y_true) != healthy
    flagged = np.asarray(y_pred) != healthy

    n_healthy = int((~truly_faulty).sum())
    n_faulty = int(truly_faulty.sum())

    far = float((flagged & ~truly_faulty).sum() / n_healthy) if n_healthy else float("nan")
    mar = float((~flagged & truly_faulty).sum() / n_faulty) if n_faulty else float("nan")
    return far, mar


def score(
    target: np.ndarray,
    predicted: np.ndarray,
    grades: Sequence[int],
    healthy: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Grading metrics, operational alarm rates, and the matrix they were derived from.

    Takes predictions rather than a model, so the same function scores the random forest
    and the convolutional network without knowing that either exists.
    """
    matrix = confusion_matrix(target, predicted, grades)
    false_alarm, missed_alarm = alarm_rates(target, predicted, healthy)
    return {
        "macro_f1": macro_f1(matrix),
        "accuracy": accuracy(matrix),
        "far": false_alarm,
        "mar": missed_alarm,
    }, matrix


def matrix_report(cm: np.ndarray, labels: Sequence[int]) -> str:
    """The confusion matrix itself, rows true and columns predicted.

    The per-class table says how well each grade does; only the matrix says which grade
    it gets mistaken for. For ordinal severities that distinction is the whole question:
    confusing neighbouring grades is a different failure from confusing the extremes.
    """
    cell = max(5, max(len(str(label)) for label in labels) + 2)
    lines = [f"{'true \\ pred':>13}" + "".join(f"{label:>{cell}}" for label in labels)]
    lines += [
        f"{label:>13}" + "".join(f"{count:>{cell}}" for count in cm[i])
        for i, label in enumerate(labels)
    ]
    return "\n".join(lines)


def report(cm: np.ndarray, labels: Sequence[int]) -> str:
    """A per-class table plus the macro average, as text."""
    precision, recall, f1 = precision_recall_f1(cm)
    support = cm.sum(axis=1)

    lines = [f"{'class':>8}  {'prec':>6}  {'recall':>6}  {'f1':>6}  {'n':>6}"]
    lines += [
        f"{label:>8}  {precision[i]:6.3f}  {recall[i]:6.3f}  {f1[i]:6.3f}  {support[i]:6d}"
        for i, label in enumerate(labels)
    ]
    lines.append(f"{'macro':>8}  {'':>6}  {'':>6}  {macro_f1(cm):6.3f}  {support.sum():6d}")
    return "\n".join(lines)
