"""Training the multi-head convolutional network and scoring it against the baseline."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from shopfloor.config import settings
from shopfloor.data import HEALTHY, PROFILE_COLUMNS
from shopfloor.metrics import matrix_report, report, score
from shopfloor.net import (
    CycleDataset,
    MultiHeadConvNet,
    channel_stats,
    count_parameters,
    get_device,
)
from shopfloor.splits import COMPONENTS, levels, split_by_run

BATCH = 64
EPOCHS = 60
LEARNING_RATE = 1e-3

N_SEGMENTS = 5
"""Time segments the trunk is pooled into. Set to 1 to reproduce global pooling.

Three stride-2 convolutions turn 600 timepoints into 75, and 75 = 3 x 5 x 5 — so the
segment count has to divide 75. Five gives 12 seconds of cycle per segment.
"""


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """One pass over the training data. Returns the mean loss of each head."""
    model.train()  # BatchNorm uses this batch's statistics, and dropout would be active
    criterion = nn.CrossEntropyLoss()
    totals = dict.fromkeys(COMPONENTS, 0.0)

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # The order below is not negotiable. PyTorch *accumulates* gradients rather than
        # overwriting them, so without zero_grad() the gradients of every batch so far
        # add up — the same effect as multiplying the learning rate by the step count.
        optimiser.zero_grad()
        outputs = model(x)
        losses = {
            component: criterion(outputs[component], y[:, i])
            for i, component in enumerate(COMPONENTS)
        }
        total = sum(losses.values())
        total.backward()
        optimiser.step()

        # .item() detaches the number from the graph; keeping the tensor would keep the
        # whole computation graph alive for as long as this dict does.
        for component, loss in losses.items():
            totals[component] += loss.item() * len(x)

    return {component: value / len(loader.dataset) for component, value in totals.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """The same losses on held-out data, with no gradients and no BatchNorm updates."""
    model.eval()  # BatchNorm switches to its running averages — silently wrong without this
    criterion = nn.CrossEntropyLoss()
    totals = dict.fromkeys(COMPONENTS, 0.0)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        for i, component in enumerate(COMPONENTS):
            totals[component] += criterion(outputs[component], y[:, i]).item() * len(x)

    return {component: value / len(loader.dataset) for component, value in totals.items()}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    """Predicted class index per component, in loader order — so shuffle must be off."""
    model.eval()
    collected: dict[str, list[np.ndarray]] = {component: [] for component in COMPONENTS}

    for x, _ in loader:
        outputs = model(x.to(device))
        for component in COMPONENTS:
            collected[component].append(outputs[component].argmax(dim=1).cpu().numpy())

    return {component: np.concatenate(parts) for component, parts in collected.items()}


def loss_line(label: str, losses: dict[str, float]) -> str:
    """Losses of all four heads on one line, so divergence between them is visible."""
    return label + "  ".join(f"{name[:5]} {value:.3f}" for name, value in losses.items())


if __name__ == "__main__":
    torch.manual_seed(settings.seed)

    root = settings.processed_dir
    tensor = np.load(root / "tensor.npy")
    labels = np.load(root / "labels.npy")

    split = split_by_run(labels, settings.split_seed)
    grades = levels(labels)
    mean, spread = channel_stats(tensor, split.train)

    train_set = CycleDataset(tensor, labels, split.train, grades, mean, spread)
    val_set = CycleDataset(tensor, labels, split.val, grades, mean, spread)
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False)

    device = get_device()
    model = MultiHeadConvNet(
        n_channels=tensor.shape[1],
        head_sizes={component: len(grades[component]) for component in COMPONENTS},
        n_segments=N_SEGMENTS,
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint = settings.models_dir / f"convnet_seg{N_SEGMENTS}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    print(f"{count_parameters(model):,} parameters over {len(train_set)} training cycles")
    print(f"device {device}, {EPOCHS} epochs, batch {BATCH}, lr {LEARNING_RATE}")
    print(f"pooling into {N_SEGMENTS} time segment(s)\n")

    for epoch in range(1, EPOCHS + 1):
        train_losses = train_epoch(model, train_loader, optimiser, device)
        val_losses = evaluate(model, val_loader, device)
        mean_val = sum(val_losses.values()) / len(val_losses)

        improved = mean_val < best
        if improved:
            best = mean_val
            # Keeping the best weights rather than the last: validation loss will start
            # rising once 30k parameters begin memorising 1586 cycles.
            torch.save(model.state_dict(), checkpoint)

        if epoch == 1 or epoch % 5 == 0:
            marker = "  saved" if improved else ""
            print(
                f"epoch {epoch:>3}  "
                + loss_line("train ", train_losses)
                + "   |   "
                + loss_line("val ", val_losses)
                + marker
            )

    print(f"\nbest mean validation loss {best:.4f}, checkpoint at {checkpoint}")

    model.load_state_dict(torch.load(checkpoint))
    predicted_indices = predict(model, val_loader, device)

    header = f"{'component':<12} {'macro F1':>9} {'accuracy':>9} {'FAR':>6} {'MAR':>6}"
    lines = [
        f"\nConv1D on validation, pooling into {N_SEGMENTS} segment(s)",
        header,
        "-" * len(header),
    ]
    matrices: list[str] = []

    for component in COMPONENTS:
        order = np.asarray(grades[component])
        # Predictions come back as 0..n-1; the same ordering that built the targets maps
        # them to physical grades, which is what the metrics and the baseline speak in.
        predicted = order[predicted_indices[component]]
        target = labels[split.val, PROFILE_COLUMNS.index(component)]

        result, matrix = score(target, predicted, grades[component], HEALTHY[component])
        lines.append(
            f"{component:<12} {result['macro_f1']:>9.3f} {result['accuracy']:>9.3f} "
            f"{result['far']:>6.3f} {result['mar']:>6.3f}"
        )
        matrices.append(
            f"\n{component}\n{report(matrix, grades[component])}\n\n"
            f"{matrix_report(matrix, grades[component])}"
        )

    summary = "\n".join(lines + matrices)
    print(summary)

    destination = settings.reports_dir / f"convnet_val_seg{N_SEGMENTS}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(summary.lstrip() + "\n")
    print(f"\nwritten to {destination}")
