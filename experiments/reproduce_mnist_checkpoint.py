#!/usr/bin/env python3
"""Train the MNIST CNN from train_and_run.py and freeze a checkpoint.

Does not import train_and_run.py (that script trains on import).
Does not run Cascade, PNR, or Experiment 1.
"""

from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cascade.shift import get_preset_spec, make_shifted_dataset

SEED = 42
EPOCHS = 5
BATCH_SIZE = 64
LR = 0.001
DATA_ROOT = os.path.join(_REPO_ROOT, "data")
OUT_DIR = os.path.join(_REPO_ROOT, "results", "reproduction_mnist")
CKPT_PATH = os.path.join(OUT_DIR, "model.pt")
METADATA_PATH = os.path.join(OUT_DIR, "metadata.json")
LOG_PATH = os.path.join(OUT_DIR, "training_log.txt")

# Existing train_and_run.py shift: aggressive preset (75° rotate + blur 7, σ=2).
# That is the repo's deterministic replacement for RandomRotation(75)+GaussianBlur(7).
SHIFT_SPEC = get_preset_spec("aggressive")


class CNN(nn.Module):
    """Same architecture as train_and_run.py."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images).argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))
    return 100.0 * correct / max(1, total)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = select_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    log_lines = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    log(f"device={device}")
    log(f"seed={SEED}")
    log(f"epochs={EPOCHS} batch_size={BATCH_SIZE} lr={LR} optimizer=Adam")

    clean_transform = T.Compose([T.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(
        root=DATA_ROOT, train=True, download=True, transform=clean_transform
    )
    test_dataset_clean = torchvision.datasets.MNIST(
        root=DATA_ROOT, train=False, download=True, transform=clean_transform
    )
    test_dataset_raw = torchvision.datasets.MNIST(
        root=DATA_ROOT, train=False, download=True, transform=None
    )
    test_dataset_shift = make_shifted_dataset(
        test_dataset_raw, SHIFT_SPEC, to_tensor=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    test_loader_clean = torch.utils.data.DataLoader(
        test_dataset_clean, batch_size=256, shuffle=False
    )
    test_loader_shift = torch.utils.data.DataLoader(
        test_dataset_shift, batch_size=256, shuffle=False
    )

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    log("Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        log(f"Epoch {epoch + 1}/{EPOCHS} — loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), CKPT_PATH)
    log(f"Saved checkpoint: {CKPT_PATH}")

    clean_acc = evaluate_accuracy(model, test_loader_clean, device)
    shifted_acc = evaluate_accuracy(model, test_loader_shift, device)
    log(f"Clean MNIST test accuracy: {clean_acc:.2f}%")
    log(f"Shifted MNIST test accuracy: {shifted_acc:.2f}%")

    gpu_name = None
    cuda_version = None
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)

    metadata = {
        "seed": SEED,
        "architecture": {
            "class": "CNN",
            "description": (
                "Conv2d(1,32,3,padding=1); ReLU; MaxPool2d(2); "
                "Conv2d(32,64,3,padding=1); ReLU; MaxPool2d(2); "
                "Linear(64*7*7,128); ReLU; Linear(128,10)"
            ),
            "matches": "train_and_run.py",
        },
        "dataset": "MNIST",
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "device": str(device),
        "pytorch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "shift": {
            "preset": "aggressive",
            "degrees": SHIFT_SPEC.degrees,
            "blur_kernel": SHIFT_SPEC.blur_kernel,
            "blur_sigma": SHIFT_SPEC.blur_sigma,
            "note": (
                "Existing train_and_run.py shift: deterministic rotate(75) + "
                "GaussianBlur(kernel=7, sigma=2.0) + ToTensor, not stochastic "
                "RandomRotation."
            ),
        },
        "clean_test_accuracy_pct": clean_acc,
        "shifted_test_accuracy_pct": shifted_acc,
        "checkpoint": CKPT_PATH,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    if not os.path.isfile(CKPT_PATH):
        raise SystemExit(f"ERROR: checkpoint missing after save: {CKPT_PATH}")
    fresh = CNN()
    state = torch.load(CKPT_PATH, map_location="cpu")
    fresh.load_state_dict(state)
    fresh.eval()
    x0, _ = test_dataset_clean[0]
    with torch.no_grad():
        logits = fresh(x0.unsqueeze(0))
    if logits.shape != (1, 10):
        raise SystemExit(f"ERROR: unexpected logit shape {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise SystemExit("ERROR: non-finite logits after reload")
    log("Checkpoint reload verification: PASSED")
    log(f"metadata: {METADATA_PATH}")
    log(f"training_log: {LOG_PATH}")


if __name__ == "__main__":
    main()
