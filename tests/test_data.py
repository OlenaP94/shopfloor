"""Tests for the sensor and profile readers."""

from pathlib import Path

import pytest

from shopfloor.data import (
    CYCLE_SECONDS,
    HEALTHY,
    PROFILE_COLUMNS,
    SENSORS,
    read_profile,
    read_sensor,
)


def test_sensor_rates_are_known() -> None:
    assert set(SENSORS.values()) == {1, 10, 100}
    assert len(SENSORS) == 17


def test_healthy_values_cover_four_components() -> None:
    assert set(HEALTHY) == {"cooler", "valve", "pump_leak", "accumulator"}
    assert set(HEALTHY) < set(PROFILE_COLUMNS)


def test_read_sensor_parses_a_matrix(tmp_path: Path) -> None:
    path = tmp_path / "PS1.txt"
    path.write_text("1.0\t2.0\t3.0\n4.5\t5.5\t6.5\n")

    matrix = read_sensor(path)

    assert matrix == [[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]]


def test_read_sensor_tolerates_mixed_whitespace(tmp_path: Path) -> None:
    path = tmp_path / "PS1.txt"
    path.write_text("1.0  2.0\t\t3.0\n")

    assert read_sensor(path) == [[1.0, 2.0, 3.0]]


def test_read_profile_names_the_columns(tmp_path: Path) -> None:
    path = tmp_path / "profile.txt"
    path.write_text("100\t100\t0\t130\t0\n3\t73\t2\t90\t1\n")

    profile = read_profile(path)

    assert profile[0] == {
        "cooler": 100,
        "valve": 100,
        "pump_leak": 0,
        "accumulator": 130,
        "stable": 0,
    }
    assert profile[1]["valve"] == 73


def test_read_profile_rejects_wrong_column_count(tmp_path: Path) -> None:
    path = tmp_path / "profile.txt"
    path.write_text("100\t100\t0\n")

    with pytest.raises(ValueError):
        read_profile(path)


def test_cycle_length_is_sixty_seconds() -> None:
    assert CYCLE_SECONDS == 60
