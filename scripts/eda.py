"""Which of the 24 channels respond to which of the four faults."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from shopfloor.arrays import TARGET_POINTS, TARGET_RATE
from shopfloor.config import settings
from shopfloor.data import HEALTHY, PROFILE_COLUMNS, Component

matplotlib.use("Agg")

FIGURES = Path("reports/figures")
TIME = np.arange(TARGET_POINTS) / TARGET_RATE
"""Seconds elapsed within a cycle, for the x axis of every waveform plot."""

NOTHING = 0.1
"""Effect sizes below this are noise; the heatmap floor, so they do not distort it."""


def load() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the processed tensor, its labels and the channel names."""
    root = settings.processed_dir
    tensor = np.load(root / "tensor.npy")
    labels = np.load(root / "labels.npy")
    names = (root / "channels.txt").read_text().split()
    return tensor, labels, names


def severity_levels(labels: np.ndarray, component: Component) -> list[int]:
    """The distinct states of one component, ordered from healthy to worst."""
    column = labels[:, PROFILE_COLUMNS.index(component)]
    return [int(v) for v in sorted(np.unique(column), key=lambda v: abs(v - HEALTHY[component]))]


def cohens_d(healthy: np.ndarray, faulty: np.ndarray) -> np.ndarray:
    """Difference between two groups in units of their spread, one value per channel."""
    pooled = np.sqrt((healthy.std(axis=0) ** 2 + faulty.std(axis=0) ** 2) / 2)
    # A channel with no spread in either group carries no information; nan keeps it
    # out of the ranking instead of turning the division into a silent infinity.
    return (healthy.mean(axis=0) - faulty.mean(axis=0)) / np.where(pooled == 0, np.nan, pooled)


def effect_sizes(summary: np.ndarray, labels: np.ndarray) -> dict[Component, np.ndarray]:
    """For every component, how far the worst cycles sit from the healthy ones."""
    scores: dict[Component, np.ndarray] = {}
    for component in HEALTHY:
        levels = severity_levels(labels, component)
        column = labels[:, PROFILE_COLUMNS.index(component)]
        healthy = summary[column == levels[0]]
        faulty = summary[column == levels[-1]]
        scores[component] = np.abs(cohens_d(healthy, faulty))
    return scores


def plot_waveforms(
    tensor: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    scores: dict[Component, np.ndarray],
) -> None:
    """One panel per component: its most responsive channel, one line per state."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, component in zip(axes.flat, HEALTHY, strict=True):
        best = int(np.nanargmax(scores[component]))
        column = labels[:, PROFILE_COLUMNS.index(component)]

        for level in severity_levels(labels, component):
            waveform = tensor[column == level, best].mean(axis=0)
            ax.plot(TIME, waveform, linewidth=1, label=str(level))

        ax.set_title(f"{component} — {names[best]}  (d = {scores[component][best]:.1f})")
        ax.set_xlabel("time within cycle, s")
        ax.legend(title="state", fontsize=8)

    fig.suptitle("Mean waveform of the most responsive channel, by component state")
    fig.tight_layout()
    fig.savefig(FIGURES / "waveforms.png", dpi=150)
    plt.close(fig)


def plot_heatmap(names: list[str], scores: dict[Component, np.ndarray]) -> None:
    """All 24 channels against all 4 components, as one picture."""
    matrix = np.column_stack([scores[component] for component in HEALTHY])

    fig, ax = plt.subplots(figsize=(6, 9))
    # Log scale, because the cooler column reaches 32 while the other three stay below
    # 1.2 — on a linear scale that one column eats the whole colour range. The floor at
    # 0.1 stops three decades of meaningless near-zero values from eating it instead.
    norm = LogNorm(vmin=NOTHING, vmax=float(np.nanmax(matrix)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", norm=norm)
    ax.set_xticks(range(len(HEALTHY)), list(HEALTHY), rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_title("Separation between healthy and worst cycles")
    fig.colorbar(image, ax=ax, label="|Cohen's d|")

    fig.tight_layout()
    fig.savefig(FIGURES / "effect_sizes.png", dpi=150)
    plt.close(fig)


def plot_rankings(names: list[str], scores: dict[Component, np.ndarray]) -> None:
    """The eight most responsive channels per component, each panel on its own scale."""
    fig, axes = plt.subplots(1, len(HEALTHY), figsize=(14, 5))

    for ax, component in zip(axes.flat, HEALTHY, strict=True):
        score = np.nan_to_num(scores[component], nan=-1.0)
        # barh draws bottom-up, so reverse once more to put the strongest on top.
        top = np.argsort(score)[::-1][:8][::-1]
        ax.barh([names[i] for i in top], score[top])
        ax.set_title(component)
        ax.set_xlabel("|Cohen's d|")
        ax.tick_params(labelsize=8)

    fig.suptitle("Eight most responsive channels per component (own scale each)")
    fig.tight_layout()
    fig.savefig(FIGURES / "rankings.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    tensor, labels, names = load()
    summary = tensor.mean(axis=2)
    scores = effect_sizes(summary, labels)

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_waveforms(tensor, labels, names, scores)
    plot_heatmap(names, scores)
    plot_rankings(names, scores)

    for component, score in scores.items():
        ranked = np.argsort(np.nan_to_num(score, nan=-1.0))[::-1][:5]
        print(f"\n{component}")
        for i in ranked:
            print(f"  {names[i]:<10} {score[i]:6.2f}")

    print(f"\nfigures written to {FIGURES}")
