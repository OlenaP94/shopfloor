"""Loading the hydraulic dataset into one numpy tensor at a common sampling rate."""

from pathlib import Path

import numpy as np

from shopfloor.config import settings
from shopfloor.data import CYCLE_SECONDS, SENSORS
from shopfloor.dataset import HydraulicDataError

TARGET_RATE = 10
"""All sensors are brought to this rate, in Hz: the middle of 1, 10 and 100."""

TARGET_POINTS = TARGET_RATE * CYCLE_SECONDS
"""Timepoints per cycle after resampling: 600."""


def read_matrix(path: str | Path) -> np.ndarray:
    """Read one tab-delimited sensor file as a float32 matrix (cycles x timepoints)."""
    return np.loadtxt(path, dtype=np.float32)


def read_labels(root: str | Path) -> np.ndarray:
    """Read profile.txt as an int32 matrix (cycles x 5 columns)."""
    return np.loadtxt(Path(root) / "profile.txt", dtype=np.int32)


def check_matrix(name: str, matrix: np.ndarray, rate: int) -> None:
    """Raise if a sensor matrix is not 2-D or has the wrong number of timepoints."""
    if matrix.ndim != 2:
        raise HydraulicDataError(f"{name}: expected a 2-D matrix, got shape {matrix.shape}")

    expected = rate * CYCLE_SECONDS
    if matrix.shape[1] != expected:
        raise HydraulicDataError(f"{name}: {matrix.shape[1]} points per cycle, expected {expected}")


def channels(name: str, matrix: np.ndarray, rate: int) -> list[tuple[str, np.ndarray]]:
    """Bring one sensor to TARGET_RATE. Fast sensors yield two channels, the others one."""
    if rate > TARGET_RATE:
        # Reshape splits every cycle into blocks of `factor` samples. The mean keeps the
        # level, the std keeps the within-block ripple that plain decimation would discard —
        # and that ripple is where a worn pump shows up first.
        factor = rate // TARGET_RATE
        blocks = matrix.reshape(len(matrix), TARGET_POINTS, factor)
        return [(f"{name}_mean", blocks.mean(axis=2)), (f"{name}_std", blocks.std(axis=2))]

    if rate < TARGET_RATE:
        # Repeat rather than interpolate: a 1 Hz sensor has one true reading per second,
        # and interpolation would invent values that were never measured.
        return [(name, np.repeat(matrix, TARGET_RATE // rate, axis=1))]

    return [(name, matrix)]


def build_tensor(root: str | Path) -> tuple[np.ndarray, list[str]]:
    """Read every sensor, resample it and stack into (cycles, channels, timepoints)."""
    root = Path(root)
    names: list[str] = []
    stack: list[np.ndarray] = []

    for name, rate in SENSORS.items():
        matrix = read_matrix(root / f"{name}.txt")
        check_matrix(name, matrix, rate)
        for channel_name, channel in channels(name, matrix, rate):
            names.append(channel_name)
            stack.append(channel)
        # A raw 100 Hz matrix is 53 MB — drop it before reading the next file, or two of
        # them are alive at once for no reason.
        del matrix

    cycles = {channel.shape[0] for channel in stack}
    if len(cycles) != 1:
        raise HydraulicDataError(f"sensors disagree on cycle count: {sorted(cycles)}")

    return np.stack(stack, axis=1), names


if __name__ == "__main__":
    tensor, names = build_tensor(settings.data_dir)
    labels = read_labels(settings.data_dir)

    print(f"tensor    {tensor.shape}  {tensor.dtype}  {tensor.nbytes / 1e6:.1f} MB")
    print(f"labels    {labels.shape}  {labels.dtype}")
    print(f"channels  {len(names)}: {', '.join(names)}")

    out = settings.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "tensor.npy", tensor)
    np.save(out / "labels.npy", labels)
    (out / "channels.txt").write_text("\n".join(names) + "\n")
    print(f"saved to {out}")
