"""Tests for Settings. Every test isolates itself from the real .env."""

import pytest
from pydantic import ValidationError

from shopfloor.config import Settings


def test_defaults_when_nothing_is_set() -> None:
    settings = Settings(_env_file=None)

    assert settings.seed == 42
    assert settings.test_size == 0.2
    assert settings.val_size == 0.15
    assert settings.anthropic_api_key is None


def test_reads_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHOPFLOOR_SEED", "123")

    assert Settings(_env_file=None).seed == 123


def test_splits_that_leave_nothing_for_training(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHOPFLOOR_TEST_SIZE", "0.9")
    monkeypatch.setenv("SHOPFLOOR_VAL_SIZE", "0.15")

    with pytest.raises(ValidationError, match="leaves nothing for training"):
        Settings(_env_file=None)


@pytest.mark.parametrize("value", ["0", "1", "1.5", "-0.1"])
def test_test_size_must_be_strictly_between_zero_and_one(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("SHOPFLOOR_TEST_SIZE", value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_seed_must_be_a_number(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHOPFLOOR_SEED", "not-a-number")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_unrelated_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHOPFLOOR_SOMETHING_ELSE", "whatever")

    assert Settings(_env_file=None).seed == 42
