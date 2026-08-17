"""Cross-checks of our metrics against scikit-learn.

test_metrics.py verifies that the code does what it was meant to do. These verify that
what it was meant to do is right — in particular the zero-division convention, which the
docstrings claim matches scikit-learn and should therefore be proved rather than asserted.
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, recall_score
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from shopfloor.metrics import (
    accuracy,
    alarm_rates,
    confusion_matrix,
    macro_f1,
    precision_recall_f1,
)

VALVE = (73, 80, 90, 100)
HEALTHY_VALVE = 100


def sample(seed: int, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Truth and predictions that agree most of the time, over unevenly frequent grades.

    A Dirichlet with concentration below 1 leaves some grades rare and occasionally
    absent, which is exactly where the two implementations could disagree.
    """
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.full(len(VALVE), 0.5))
    y_true = rng.choice(VALVE, size=n, p=weights)
    y_pred = np.where(rng.random(n) < 0.7, y_true, rng.choice(VALVE, size=n))
    return y_true, y_pred


@pytest.mark.parametrize("seed", range(10))
def test_confusion_matrix_matches_sklearn(seed: int) -> None:
    y_true, y_pred = sample(seed)

    ours = confusion_matrix(y_true, y_pred, VALVE)
    theirs = sklearn_confusion_matrix(y_true, y_pred, labels=VALVE)
    assert np.array_equal(ours, theirs)


@pytest.mark.parametrize("seed", range(10))
def test_precision_recall_f1_match_sklearn(seed: int) -> None:
    y_true, y_pred = sample(seed)

    ours = precision_recall_f1(confusion_matrix(y_true, y_pred, VALVE))
    theirs = precision_recall_fscore_support(y_true, y_pred, labels=VALVE, zero_division=0)[:3]

    for mine, theirs_ in zip(ours, theirs, strict=True):
        assert np.allclose(mine, theirs_)


@pytest.mark.parametrize("seed", range(10))
def test_macro_f1_and_accuracy_match_sklearn(seed: int) -> None:
    y_true, y_pred = sample(seed)
    cm = confusion_matrix(y_true, y_pred, VALVE)

    assert macro_f1(cm) == pytest.approx(
        f1_score(y_true, y_pred, labels=VALVE, average="macro", zero_division=0)
    )
    assert accuracy(cm) == pytest.approx(accuracy_score(y_true, y_pred))


@pytest.mark.parametrize("seed", range(10))
def test_alarm_rates_match_a_binarised_sklearn_matrix(seed: int) -> None:
    y_true, y_pred = sample(seed)
    far, mar = alarm_rates(y_true, y_pred, HEALTHY_VALVE)

    faulty_true = (y_true != HEALTHY_VALVE).astype(int)
    faulty_pred = (y_pred != HEALTHY_VALVE).astype(int)
    true_negative, false_positive = sklearn_confusion_matrix(
        faulty_true, faulty_pred, labels=[0, 1]
    )[0]

    # Some seeds draw no healthy cycles at all, and then a false-alarm rate has no meaning.
    if true_negative + false_positive:
        assert far == pytest.approx(false_positive / (true_negative + false_positive))
    else:
        assert np.isnan(far)

    if faulty_true.sum():
        assert mar == pytest.approx(1 - recall_score(faulty_true, faulty_pred, zero_division=0))
    else:
        assert np.isnan(mar)


def test_the_zero_division_convention_matches_sklearn() -> None:
    """Grades 73 and 80 occur nowhere, so their precision and recall are both 0/0."""
    y_true = np.array([100, 100, 90])
    y_pred = np.array([100, 90, 90])

    ours = precision_recall_f1(confusion_matrix(y_true, y_pred, VALVE))
    theirs = precision_recall_fscore_support(y_true, y_pred, labels=VALVE, zero_division=0)[:3]

    for mine, theirs_ in zip(ours, theirs, strict=True):
        assert np.allclose(mine, theirs_)
        assert mine[0] == 0.0 and mine[1] == 0.0


def test_a_perfect_and_a_hopeless_model_both_agree_with_sklearn() -> None:
    y_true = np.array(list(VALVE) * 5)

    for y_pred in (y_true, np.full_like(y_true, HEALTHY_VALVE)):
        cm = confusion_matrix(y_true, y_pred, VALVE)
        assert macro_f1(cm) == pytest.approx(
            f1_score(y_true, y_pred, labels=VALVE, average="macro", zero_division=0)
        )
