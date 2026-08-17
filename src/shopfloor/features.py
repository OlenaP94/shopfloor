"""Turning the raw tensor into a flat table of per-window features."""

import numpy as np

from shopfloor.config import settings

N_WINDOWS = 6
"""Windows per 60 s cycle. Six keeps 864 features well below the 2205 cycles."""

STATISTICS = ("mean", "std", "min", "max", "p25", "p75")
"""Computed for every channel in every window, in this order."""

VIRTUAL = ("CE", "CP", "SE")
"""Channels the rig computes rather than measures — see the README."""


def window_features(tensor: np.ndarray, n_windows: int = N_WINDOWS) -> np.ndarray:
    """Summarise each window of each channel: (cycles, channels, points) -> (cycles, columns)."""
    n_cycles, n_channels, n_points = tensor.shape
    if n_points % n_windows:
        raise ValueError(f"{n_points} timepoints do not divide into {n_windows} windows")

    windows = tensor.reshape(n_cycles, n_channels, n_windows, n_points // n_windows)
    parts = [
        windows.mean(axis=3),
        windows.std(axis=3),
        windows.min(axis=3),
        windows.max(axis=3),
        np.percentile(windows, 25, axis=3),
        np.percentile(windows, 75, axis=3),
    ]

    # (cycles, channels, windows, statistics), then flattened channel-major so that
    # the column order matches feature_names exactly. Nothing but this comment and
    # the tests keeps the two in step, so they are tested together.
    stacked = np.stack(parts, axis=3)
    return stacked.reshape(n_cycles, -1).astype(np.float32)


def feature_names(channels: list[str], n_windows: int = N_WINDOWS) -> list[str]:
    """Column names, in the same order window_features produces the columns."""
    return [
        f"{channel}_w{window}_{statistic}"
        for channel in channels
        for window in range(n_windows)
        for statistic in STATISTICS
    ]


def virtual_mask(channels: list[str], n_windows: int = N_WINDOWS) -> np.ndarray:
    """True for every column derived from a channel the rig computed rather than measured."""
    per_channel = np.array([channel in VIRTUAL for channel in channels])
    return np.repeat(per_channel, n_windows * len(STATISTICS))


if __name__ == "__main__":
    root = settings.processed_dir
    tensor = np.load(root / "tensor.npy")
    channels = (root / "channels.txt").read_text().split()

    features = window_features(tensor)
    names = feature_names(channels)
    if features.shape[1] != len(names):
        raise ValueError(f"{features.shape[1]} columns but {len(names)} names")

    mask = virtual_mask(channels)
    print(f"features  {features.shape}  {features.dtype}  {features.nbytes / 1e6:.1f} MB")
    print(f"virtual   {mask.sum()} of {len(names)} columns")

    np.save(root / "features.npy", features)
    (root / "feature_names.txt").write_text("\n".join(names) + "\n")
    print(f"saved to {root}")
