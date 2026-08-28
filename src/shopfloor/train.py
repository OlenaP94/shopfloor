"""Training the multi-head convolutional network and scoring it against the baseline."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from shopfloor.config import settings
from shopfloor.data import HEALTHY, PROFILE_COLUMNS
from shopfloor.metrics import confusion_matrix, macro_f1, matrix_report, report, score
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

LOSS_WEIGHTS: dict[str, float] = {
    "cooler": 1.0,
    "valve": 1.0,
    "pump_leak": 1.0,
    "accumulator": 1.0,
}
"""How much each head's loss counts towards the shared trunk's gradient.

Kept at 1.0 after testing 3.0 on the accumulator. That raised its macro F1 by 0.005 —
far below the 0.05 swing between neighbouring epochs of a single run, so no effect was
detected — while doubling its missed-alarm rate from 0.059 to 0.108, because the model
traded misses for false alarms. Worse where it matters, no better where it is measured.
"""

RUN = f"seg{N_SEGMENTS}_w{LOSS_WEIGHTS['accumulator']:g}"
"""Names the checkpoint and report, so runs do not overwrite each other."""


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    weights: dict[str, float],
) -> dict[str, float]:
    """One pass over the training data. Returns the mean *unweighted* loss of each head.

    The weights change what the optimiser chases; the reported losses stay unweighted so
    that numbers from differently weighted runs remain comparable.
    """
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
        weighted = sum(weights[component] * losses[component] for component in COMPONENTS)
        weighted.backward()
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


def validation_macro_f1(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    targets: dict[str, np.ndarray],
    grades: dict[str, list[int]],
) -> dict[str, float]:
    """Macro F1 per component on the validation split.

    Selecting the checkpoint on this rather than on the mean loss, because the four
    losses are not on a comparable scale: the cooler reaches 0.000 by epoch 25 and then
    decides the mean, which leaves the accumulator's weights chosen essentially at random.
    Choosing on the metric that gets reported keeps selection and evaluation in one
    language.
    """
    predicted_indices = predict(model, loader, device)
    result: dict[str, float] = {}

    for component in COMPONENTS:
        order = np.asarray(grades[component])
        matrix = confusion_matrix(
            targets[component], order[predicted_indices[component]], grades[component]
        )
        result[component] = macro_f1(matrix)
    return result


def metric_line(label: str, values: dict[str, float]) -> str:
    """All four heads on one line, so divergence between them stays visible."""
    return label + "  ".join(f"{name[:5]} {value:.3f}" for name, value in values.items())


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

    targets = {
        component: labels[split.val, PROFILE_COLUMNS.index(component)] for component in COMPONENTS
    }

    checkpoint = settings.models_dir / f"convnet_{RUN}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch = 0.0, 0

    print(f"{count_parameters(model):,} parameters over {len(train_set)} training cycles")
    print(f"device {device}, {EPOCHS} epochs, batch {BATCH}, lr {LEARNING_RATE}")
    print(f"pooling into {N_SEGMENTS} time segment(s)")
    print(f"loss weights {LOSS_WEIGHTS}\n")

    for epoch in range(1, EPOCHS + 1):
        train_losses = train_epoch(model, train_loader, optimiser, device, LOSS_WEIGHTS)
        val_losses = evaluate(model, val_loader, device)
        val_f1 = validation_macro_f1(model, val_loader, device, targets, grades)
        mean_f1 = sum(val_f1.values()) / len(val_f1)

        improved = mean_f1 > best
        if improved:
            best, best_epoch = mean_f1, epoch
            # Best weights, not last: validation stops improving long before epoch 60,
            # once 39k parameters begin memorising 1586 cycles.
            torch.save(model.state_dict(), checkpoint)

        if epoch == 1 or epoch % 5 == 0:
            gap = sum(val_losses.values()) - sum(train_losses.values())
            print(f"epoch {epoch:>3}  " + metric_line("train loss  ", train_losses))
            print(
                " " * 10
                + metric_line("val F1      ", val_f1)
                + f"   mean {mean_f1:.3f}   gap {gap:+.2f}"
                + ("  saved" if improved else "")
            )

    print(f"\nbest mean macro F1 {best:.4f} at epoch {best_epoch}, checkpoint {checkpoint}")

    model.load_state_dict(torch.load(checkpoint))
    predicted_indices = predict(model, val_loader, device)

    header = f"{'component':<12} {'macro F1':>9} {'accuracy':>9} {'FAR':>6} {'MAR':>6}"
    lines = [
        f"\nConv1D on validation — {N_SEGMENTS} segment(s), "
        f"accumulator loss weight {LOSS_WEIGHTS['accumulator']:g}",
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

    destination = settings.reports_dir / f"convnet_val_{RUN}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(summary.lstrip() + "\n")
    print(f"\nwritten to {destination}")
