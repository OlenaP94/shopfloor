"""Tests for HydraulicDataset, built on a miniature fake dataset."""

from pathlib import Path

import pytest

from shopfloor.dataset import HydraulicDataError, HydraulicDataset


def test_length_comes_from_the_profile(tiny_dataset: Path) -> None:
    assert len(HydraulicDataset(tiny_dataset)) == 3


def test_getitem_returns_signals_and_labels(tiny_dataset: Path) -> None:
    cycle = HydraulicDataset(tiny_dataset)[0]

    assert cycle["labels"]["cooler"] == 100
    assert len(cycle["signals"]) == 17
    assert len(cycle["signals"]["PS1"]) == 6000
    assert len(cycle["signals"]["TS1"]) == 60


def test_dataset_is_iterable(tiny_dataset: Path) -> None:
    assert sum(1 for _ in HydraulicDataset(tiny_dataset)) == 3


@pytest.mark.parametrize(
    ("component", "expected"),
    [
        ("cooler", [0, 2]),
        ("valve", [0]),
        ("pump_leak", [0]),
        ("accumulator", [0]),
    ],
)
def test_healthy_indices_per_component(
    tiny_dataset: Path, component: str, expected: list[int]
) -> None:
    assert HydraulicDataset(tiny_dataset).healthy_indices(component) == expected


def test_healthy_in_all_components(tiny_dataset: Path) -> None:
    assert HydraulicDataset(tiny_dataset).healthy_indices() == [0]


def test_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(HydraulicDataError, match="run `make data`"):
        HydraulicDataset(tmp_path / "nowhere")


def test_missing_sensor_file(tiny_dataset: Path) -> None:
    (tiny_dataset / "PS1.txt").unlink()

    with pytest.raises(HydraulicDataError, match="missing files"):
        HydraulicDataset(tiny_dataset)


def test_empty_profile(tiny_dataset: Path) -> None:
    (tiny_dataset / "profile.txt").write_text("")

    with pytest.raises(HydraulicDataError, match="no rows"):
        HydraulicDataset(tiny_dataset)


def test_sensor_with_wrong_cycle_count(tiny_dataset: Path) -> None:
    row = "\t".join(str(float(i)) for i in range(6000))
    (tiny_dataset / "PS1.txt").write_text(row + "\n")  # one cycle instead of three

    with pytest.raises(HydraulicDataError, match="expected 3"):
        HydraulicDataset(tiny_dataset)


def test_sensor_with_wrong_point_count(tiny_dataset: Path) -> None:
    row = "\t".join(str(float(i)) for i in range(99))  # should be 6000
    (tiny_dataset / "PS1.txt").write_text("\n".join([row] * 3) + "\n")

    with pytest.raises(HydraulicDataError, match="expected 6000"):
        HydraulicDataset(tiny_dataset)


def test_unknown_component(tiny_dataset: Path) -> None:
    with pytest.raises(HydraulicDataError, match="unknown component"):
        HydraulicDataset(tiny_dataset).healthy_indices("gearbox")


def test_repr_mentions_size(tiny_dataset: Path) -> None:
    text = repr(HydraulicDataset(tiny_dataset))

    assert "3 cycles" in text
    assert "17 sensors" in text


def test_labels_returns_one_value_per_cycle(tiny_dataset: Path) -> None:
    assert HydraulicDataset(tiny_dataset).labels("valve") == [100, 73, 90]


def test_labels_rejects_unknown_column(tiny_dataset: Path) -> None:
    with pytest.raises(HydraulicDataError, match="unknown column"):
        HydraulicDataset(tiny_dataset).labels("gearbox")
