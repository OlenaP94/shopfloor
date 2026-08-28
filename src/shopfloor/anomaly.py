"""Reconstruction-based anomaly detection: the layer that answers "this is unfamiliar".

The classifier knows four components and nothing else. Shown a burst hose it will still
name a valve grade, because that is all it can say. An autoencoder trained on one known
condition answers a different question — whether the input belongs to the population it
was trained on at all.

What can honestly be demonstrated here is narrower than "detects unknown faults", because
the dataset contains no fault outside the four labelled components. Training on cycles
with a healthy cooler and measuring reconstruction error against the degraded ones shows
whether the error tracks severity **without ever seeing a label**. That is the claim.

Note also that "cooler = 100" is not "machine healthy": the valve, pump and accumulator
vary freely across those cycles.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from shopfloor.config import settings
from shopfloor.data import HEALTHY, PROFILE_COLUMNS
from shopfloor.net import CycleDataset, channel_stats, count_parameters, get_device
from shopfloor.splits import COMPONENTS, levels, split_by_run

BATCH = 32
EPOCHS = 80
LEARNING_RATE = 1e-3
HEALTHY_COOLER = 100
FLAG_PERCENTILE = 99.0
"""Cycles reconstructed worse than this percentile of the training errors are flagged."""


class ConvAutoencoder(nn.Module):
    """Encoder down to 16 channels x 75 timepoints, decoder back to the original shape.

    The bottleneck is the whole mechanism: 24 x 600 values have to pass through 16 x 75,
    a factor of twelve. The network can only afford to keep structure it has seen often,
    so anything unfamiliar comes back distorted.
    """

    def __init__(self, n_channels: int, width: int = 32, bottleneck: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, width, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(width, bottleneck, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                bottleneck, width, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.ConvTranspose1d(width, width, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.ConvTranspose1d(
                width, n_channels, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One pass over the familiar cycles. Returns the mean reconstruction loss."""
    model.train()
    criterion = nn.MSELoss()
    total = 0.0

    for x, _ in loader:  # the labels exist but this model never sees them
        x = x.to(device)
        optimiser.zero_grad()
        loss = criterion(model(x), x)
        loss.backward()
        optimiser.step()
        total += loss.item() * len(x)

    return total / len(loader.dataset)


@torch.no_grad()
def reconstruction_error(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Mean squared error per cycle, in loader order — so shuffle must be off."""
    model.eval()
    errors: list[np.ndarray] = []

    for x, _ in loader:
        x = x.to(device)
        per_cycle = ((model(x) - x) ** 2).mean(dim=(1, 2))
        errors.append(per_cycle.cpu().numpy())

    return np.concatenate(errors)


if __name__ == "__main__":
    torch.manual_seed(settings.seed)

    root = settings.processed_dir
    tensor = np.load(root / "tensor.npy")
    labels = np.load(root / "labels.npy")

    split = split_by_run(labels, settings.split_seed)
    grades = levels(labels)
    mean, spread = channel_stats(tensor, split.train)

    cooler = labels[:, PROFILE_COLUMNS.index("cooler")]
    familiar = split.train & (cooler == HEALTHY_COOLER)

    # Same scaling as the classifier, fitted on the same training rows, so the two models
    # see identical numbers and their errors stay comparable.
    familiar_set = CycleDataset(tensor, labels, familiar, grades, mean, spread)
    val_set = CycleDataset(tensor, labels, split.val, grades, mean, spread)
    train_loader = DataLoader(familiar_set, batch_size=BATCH, shuffle=True)
    familiar_eval = DataLoader(familiar_set, batch_size=BATCH, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False)

    device = get_device()
    model = ConvAutoencoder(n_channels=tensor.shape[1]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"{count_parameters(model):,} parameters")
    print(f"training on {int(familiar.sum())} cycles with cooler = {HEALTHY_COOLER}")
    print(f"device {device}, {EPOCHS} epochs, batch {BATCH}\n")

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimiser, device)
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:>3}  reconstruction loss {loss:.5f}")

    checkpoint = settings.models_dir / "autoencoder.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)

    # The threshold comes from the familiar cycles alone: the detector is not allowed to
    # look at degraded data, or it stops being unsupervised.
    familiar_errors = reconstruction_error(model, familiar_eval, device)
    threshold = float(np.percentile(familiar_errors, FLAG_PERCENTILE))

    val_errors = reconstruction_error(model, val_loader, device)
    val_cooler = cooler[split.val]

    lines = [
        f"\nAutoencoder trained on cooler = {HEALTHY_COOLER} only",
        f"threshold = {FLAG_PERCENTILE:g}th percentile of training error = {threshold:.5f}",
        "",
        f"{'cooler':>8} {'cycles':>8} {'mean error':>12} {'median':>10} {'flagged':>9}",
        "-" * 52,
    ]

    for grade in grades["cooler"]:
        selected = val_errors[val_cooler == grade]
        flagged = float((selected > threshold).mean())
        lines.append(
            f"{grade:>8} {len(selected):>8} {selected.mean():>12.5f} "
            f"{np.median(selected):>10.5f} {flagged:>8.1%}"
        )

    # The control that decides what this result is worth. The cooler was held healthy in
    # training, so its degraded grades are unfamiliar and should be flagged. The other
    # three varied freely, so the detector has seen every one of their grades and must
    # NOT separate them — otherwise it is not detecting unfamiliarity, it is detecting
    # "something is wrong", which no amount of reconstruction error can justify.
    lines += [
        "",
        f"{'component':>12} {'grades seen in training':<28} {'AUC':>6}  interpretation",
        "-" * 72,
    ]

    for component in COMPONENTS:
        column = labels[:, PROFILE_COLUMNS.index(component)]
        seen = sorted(set(column[familiar].tolist()))
        auc = roc_auc_score((column[split.val] != HEALTHY[component]).astype(int), val_errors)
        note = "held out — should be flagged" if len(seen) == 1 else "seen — should not separate"
        lines.append(f"{component:>12} {str(seen):<28} {auc:>6.3f}  {note}")

    lines += [
        "",
        "No labels were used in training. A high AUC on the held-out component together",
        "with AUC near 0.5 on the others is the result worth having: the detector reacts",
        "to conditions absent from its training data, and only to those.",
    ]

    summary = "\n".join(lines)
    print(summary)

    destination = settings.reports_dir / "autoencoder_val.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(summary.lstrip() + "\n")
    print(f"\nwritten to {destination}")
