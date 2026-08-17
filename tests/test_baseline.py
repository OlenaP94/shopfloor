"""Tests for the random forest baselines."""

import numpy as np

from shopfloor.baseline import fit, normalise, score, top_features

GRADES = [0, 1]


def test_normalise_standardises_the_training_rows() -> None:
    features = np.array([[10.0], [20.0], [30.0], [40.0]])
    train = np.array([True, True, False, False])

    scaled = normalise(features, train)
    assert scaled[train].mean() == 0.0
    assert scaled[train].std() == 1.0


def test_the_untrained_rows_are_transformed_with_the_training_statistics() -> None:
    """Validation must be rescaled by numbers it did not contribute to."""
    features = np.array([[10.0], [20.0], [30.0]])
    train = np.array([True, True, False])

    scaled = normalise(features, train)
    mean, spread = 15.0, 5.0  # from the first two rows only
    assert scaled[2, 0] == np.float32((30.0 - mean) / spread)
    assert scaled[2, 0] > 1.0  # and so it lands outside the training range


def test_a_constant_column_becomes_zero_rather_than_infinity() -> None:
    features = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    train = np.ones(3, dtype=bool)

    scaled = normalise(features, train)
    assert np.all(scaled[:, 1] == 0.0)
    assert np.isfinite(scaled).all()


def test_a_forest_learns_a_separable_signal() -> None:
    rng = np.random.default_rng(0)
    target = np.array([0] * 40 + [1] * 40)
    features = rng.normal(0.0, 0.1, size=(80, 3))
    features[:, 0] += target * 5.0  # column 0 alone separates the classes

    part = np.zeros(80, dtype=bool)
    part[::2] = True  # every other cycle trains, the rest is held out

    forest = fit(features, target, part, seed=0)
    result, matrix = score(target[~part], forest.predict(features[~part]), GRADES, healthy=0)

    assert result["macro_f1"] == 1.0
    assert result["far"] == 0.0
    assert result["mar"] == 0.0
    assert matrix.trace() == matrix.sum()  # everything on the diagonal


def test_the_forest_names_the_column_that_carries_the_signal() -> None:
    rng = np.random.default_rng(0)
    target = np.array([0] * 40 + [1] * 40)
    features = rng.normal(0.0, 0.1, size=(80, 3))
    features[:, 2] += target * 5.0

    forest = fit(features, target, np.ones(80, dtype=bool), seed=0)
    assert top_features(forest, ["noise_a", "noise_b", "signal"], count=1) == ["signal"]


def test_a_forest_given_only_noise_scores_far_below_one() -> None:
    """The floor of the floor: no signal, so macro F1 must not look like success."""
    rng = np.random.default_rng(0)
    target = np.array([0, 1] * 40)
    features = rng.normal(size=(80, 5))

    part = np.zeros(80, dtype=bool)
    part[:40] = True

    forest = fit(features, target, part, seed=0)
    result, _ = score(target[~part], forest.predict(features[~part]), GRADES, healthy=0)
    assert result["macro_f1"] < 0.75
