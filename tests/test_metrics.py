"""Tests for the evaluation metrics."""

import numpy as np
import pytest

from shopfloor.metrics import (
    accuracy,
    alarm_rates,
    confusion_matrix,
    macro_f1,
    matrix_report,
    precision_recall_f1,
    report,
)

COOLER = (100, 20, 3)


def test_perfect_predictions_score_one() -> None:
    y = np.array([100, 100, 20, 3, 3])
    cm = confusion_matrix(y, y, COOLER)

    precision, recall, f1 = precision_recall_f1(cm)
    assert np.allclose(precision, 1.0)
    assert np.allclose(recall, 1.0)
    assert np.allclose(f1, 1.0)
    assert macro_f1(cm) == 1.0
    assert accuracy(cm) == 1.0
    assert alarm_rates(y, y, healthy=100) == (0.0, 0.0)


def test_rows_are_truth_and_columns_are_predictions() -> None:
    """Two cycles that were healthy and got called broken land in row 0, column 2."""
    y_true = np.array([100, 100])
    y_pred = np.array([3, 3])
    cm = confusion_matrix(y_true, y_pred, COOLER)

    assert cm[0, 2] == 2
    assert cm.sum() == 2


def test_counts_from_a_hand_built_matrix() -> None:
    cm = np.array([[700, 35, 6], [28, 680, 24], [4, 18, 710]])
    precision, recall, f1 = precision_recall_f1(cm)

    # class 20: 680 correct, 35 + 18 wrongly called 20, 28 + 24 missed
    assert precision[1] == pytest.approx(680 / 733, abs=1e-6)
    assert recall[1] == pytest.approx(680 / 732, abs=1e-6)
    assert f1[1] == pytest.approx(2 * precision[1] * recall[1] / (precision[1] + recall[1]))


def test_a_class_that_never_appears_scores_zero_not_nan() -> None:
    y = np.array([100, 100, 20])
    cm = confusion_matrix(y, y, COOLER)  # class 3 occurs neither as truth nor prediction

    precision, recall, f1 = precision_recall_f1(cm)
    assert (precision[2], recall[2], f1[2]) == (0.0, 0.0, 0.0)
    assert not np.isnan(f1).any()


def test_accuracy_can_look_good_while_macro_f1_exposes_the_failure() -> None:
    """A model that always says "healthy" on imbalanced data — the whole reason for macro F1."""
    y_true = np.array([100] * 95 + [3] * 5)
    y_pred = np.array([100] * 100)
    cm = confusion_matrix(y_true, y_pred, COOLER)

    assert accuracy(cm) == pytest.approx(0.95)
    assert macro_f1(cm) < 0.5

    far, mar = alarm_rates(y_true, y_pred, healthy=100)
    assert far == 0.0  # it never cries wolf
    assert mar == 1.0  # because it never cries at all


def test_a_model_that_flags_everything_has_no_misses_and_every_false_alarm() -> None:
    y_true = np.array([100] * 10 + [20] * 10)
    y_pred = np.array([3] * 20)

    far, mar = alarm_rates(y_true, y_pred, healthy=100)
    assert (far, mar) == (1.0, 0.0)


def test_wrong_severity_is_not_a_missed_alarm() -> None:
    """Grading a broken cooler at 20 instead of 3 still raises the alarm."""
    y_true = np.array([3, 3, 3])
    y_pred = np.array([20, 20, 20])

    far, mar = alarm_rates(y_true, y_pred, healthy=100)
    assert mar == 0.0

    cm = confusion_matrix(y_true, y_pred, COOLER)
    assert macro_f1(cm) == 0.0  # but the grading itself is entirely wrong


def test_an_alarm_rate_with_nothing_to_measure_is_nan_not_zero() -> None:
    """No healthy cycles means no false alarms were possible, which is not the same as none."""
    all_faulty = np.array([3, 3, 20])
    far, mar = alarm_rates(all_faulty, all_faulty, healthy=100)
    assert np.isnan(far)
    assert mar == 0.0

    all_healthy = np.array([100, 100])
    far, mar = alarm_rates(all_healthy, all_healthy, healthy=100)
    assert far == 0.0
    assert np.isnan(mar)


def test_unknown_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="not in labels"):
        confusion_matrix(np.array([100, 55]), np.array([100, 100]), COOLER)


def test_matrix_report_puts_truth_in_rows_and_predictions_in_columns() -> None:
    """Two healthy coolers called broken must appear on the 100 row, in the 3 column."""
    y_true = np.array([100, 100, 20])
    y_pred = np.array([3, 3, 20])
    text = matrix_report(confusion_matrix(y_true, y_pred, COOLER), COOLER)

    rows = text.splitlines()
    assert rows[0].split() == ["true", "\\", "pred", "100", "20", "3"]
    assert rows[1].split() == ["100", "0", "0", "2"]
    assert rows[2].split() == ["20", "0", "1", "0"]
    assert len(rows) == len(COOLER) + 1


def test_report_lists_every_class_and_the_macro_row() -> None:
    y = np.array([100, 20, 3])
    text = report(confusion_matrix(y, y, COOLER), COOLER)

    assert len(text.splitlines()) == len(COOLER) + 2  # header + one row per class + macro
    assert "macro" in text
