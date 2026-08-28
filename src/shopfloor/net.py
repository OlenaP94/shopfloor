"""The convolutional model: device, dataset and architecture."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from shopfloor.data import PROFILE_COLUMNS, Component
from shopfloor.splits import COMPONENTS


def get_device() -> torch.device:
    """The best available device: Apple Silicon GPU, then CUDA, then CPU.

    On this machine the answer is "mps", not "cuda" — half the tutorials online hardcode
    cuda and fail on the first .to() call. Asking once, here, keeps that out of the model.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def channel_stats(tensor: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and spread over the training cycles, shaped to broadcast.

    Statistics are per channel, not per column: a channel is one physical quantity and
    keeps its identity across the whole cycle. Taken over training rows only — the same
    rule as everywhere else, and the reason this is a function rather than one line.
    """
    training = tensor[train]
    mean = training.mean(axis=(0, 2)).reshape(-1, 1)
    spread = training.std(axis=(0, 2)).reshape(-1, 1)
    return mean, np.where(spread == 0, 1.0, spread)


class CycleDataset(Dataset):
    """One split of cycles: a (channels, timepoints) window and four class indices.

    Cross-entropy wants indices 0..n-1, but the labels are physical grades — 100, 20, 3
    for the cooler. The mapping follows the order in `grades`, so the same list must be
    used to translate predictions back.
    """

    def __init__(
        self,
        tensor: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        grades: dict[Component, list[int]],
        mean: np.ndarray,
        spread: np.ndarray,
    ) -> None:
        scaled = (tensor[mask] - mean) / spread
        self.x = torch.from_numpy(scaled.astype(np.float32))

        position = {
            component: {grade: i for i, grade in enumerate(grades[component])}
            for component in COMPONENTS
        }
        self.y = torch.tensor(
            [
                [position[c][int(row[PROFILE_COLUMNS.index(c)])] for c in COMPONENTS]
                for row in labels[mask]
            ],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class MultiHeadConvNet(nn.Module):
    """One shared convolutional trunk, one small classification head per component.

    The trunk learns what a hydraulic signal looks like; each head decides one component.
    That is the bet of multi-task learning — and its risk, since the heads will not learn
    at the same rate and a strong one can drown a weak one.
    """

    def __init__(
        self,
        n_channels: int,
        head_sizes: dict[str, int],
        width: int = 64,
        n_segments: int = 6,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, width, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        # Pooling into n_segments instead of into a single value keeps *when* a feature
        # fired. A cycle is a fixed 60-second program, so "at second 20" carries
        # information — the forest had it explicitly in names like PS2_mean_w0_std, and
        # global pooling throws it away. n_segments=1 reproduces the global behaviour.
        self.n_segments = n_segments
        self.heads = nn.ModuleDict(
            {name: nn.Linear(2 * width * n_segments, size) for name, size in head_sizes.items()}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        activations = self.features(x)
        batch, width, length = activations.shape

        if length % self.n_segments:
            raise ValueError(
                f"trunk output length {length} does not divide into {self.n_segments} segments"
            )

        # The same reshape used to resample the sensors and to build the windowed
        # features: split an axis into blocks, then aggregate along the last one. Written
        # out rather than using AdaptiveAvgPool1d, which on MPS refuses lengths that do
        # not divide evenly — and which hides what is actually happening.
        blocks = activations.reshape(batch, width, self.n_segments, length // self.n_segments)

        # Both statistics per segment: the average says "how much of this feature here",
        # the maximum says "this pattern occurred here". For a two-second transient in a
        # sixty-second cycle the maximum is the one that survives.
        pooled = torch.cat([blocks.mean(dim=3).flatten(1), blocks.amax(dim=3).flatten(1)], dim=1)
        return {name: head(pooled) for name, head in self.heads.items()}


def count_parameters(model: nn.Module) -> int:
    """Trainable parameters, to be compared against the number of training cycles."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from shopfloor.config import settings
    from shopfloor.splits import levels, split_by_run

    root = settings.processed_dir
    tensor = np.load(root / "tensor.npy")
    labels = np.load(root / "labels.npy")

    split = split_by_run(labels, settings.split_seed)
    grades = levels(labels)
    mean, spread = channel_stats(tensor, split.train)

    train_set = CycleDataset(tensor, labels, split.train, grades, mean, spread)
    loader = DataLoader(train_set, batch_size=64, shuffle=True)
    x, y = next(iter(loader))

    device = get_device()
    model = MultiHeadConvNet(
        n_channels=tensor.shape[1],
        head_sizes={component: len(grades[component]) for component in COMPONENTS},
    ).to(device)
    outputs = model(x.to(device))

    print(f"device        {device}")
    print(f"train cycles  {len(train_set)}")
    print(f"batch x       {tuple(x.shape)}  {x.dtype}")
    print(f"batch y       {tuple(y.shape)}  {y.dtype}  first row {y[0].tolist()}")
    print(f"parameters    {count_parameters(model):,}")
    for name, logits in outputs.items():
        print(f"head {name:<12} {tuple(logits.shape)}")
