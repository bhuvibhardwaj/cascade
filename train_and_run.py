"""
MNIST CNN experiment — hardened version.

Fixes vs the original train_and_run.py:
  1. Clean/shifted pairing uses dataset_idx (not position in misclassified list).
  2. Shift is genuinely deterministic (T.functional.rotate + explicit sigma).
  3. Samples are structured dataclasses with index + prediction metadata.
  4. Seeds everywhere (Python / NumPy / PyTorch), recorded in config.json.
  5. Saves model checkpoint + every experiment artifact.
  6. Four-group analysis: shifted-correct AND shifted-incorrect, not just the latter.
  7. Reproducible sample selection (fixed RNG seed, not first-N).
  8. Produces plots + CSV tables, not just stdout.

Run: python3 train_and_run.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import random
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from cascade import (
    Cascade,
    Sample,
    ShiftSpec,
    build_shift,
    calibrate_pnr_thresholds,
    diagnose,
    fragility_profile,
    get_preset_spec,
    make_shifted_dataset,
)
from cascade.diagnose import DiagnosisResult
from cascade.plots import save_layer_profiles, save_summary_json

# ---------------------------------------------------------------------------
# 0. Config + seeds
# ---------------------------------------------------------------------------

SEED = 42
EXPERIMENT_NAME = "mnist_cnn"
SHIFT_NAME = "aggressive"
SHIFT_SPEC = get_preset_spec(SHIFT_NAME)
N_TRAIN_EPOCHS = 5
N_SAMPLES_PROFILE = 200
PNR_QUANTILE = 0.95

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# Deterministic ops where possible
torch.use_deterministic_algorithms(False)  # some ops (eg rotate) aren't fully deterministic on MPS/CUDA

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

OUT_ROOT = os.path.join(
    "results", f"{EXPERIMENT_NAME}_{SHIFT_NAME}_seed{SEED}_{int(time.time())}"
)
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "plots"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUT_ROOT, "run.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("mnist")
log.info(f"Device: {device}")
log.info(f"Output root: {OUT_ROOT}")


# ---------------------------------------------------------------------------
# 1. Data — clean dataset + genuinely-deterministic shifted dataset wrapper
# ---------------------------------------------------------------------------

clean_transform = T.Compose([T.ToTensor()])

train_dataset_raw = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=None
)
test_dataset_raw = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=None
)

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=clean_transform
)
test_dataset_clean = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=clean_transform
)
test_dataset_shift = make_shifted_dataset(test_dataset_raw, SHIFT_SPEC, to_tensor=True)

BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)
test_loader_clean = torch.utils.data.DataLoader(
    test_dataset_clean, batch_size=1, shuffle=False
)
test_loader_shift = torch.utils.data.DataLoader(
    test_dataset_shift, batch_size=1, shuffle=False
)


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------

class CNN(nn.Module):
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


model = CNN().to(device)


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

log.info(f"Training for {N_TRAIN_EPOCHS} epochs...")
for epoch in range(N_TRAIN_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    log.info(f"Epoch {epoch + 1}/{N_TRAIN_EPOCHS} — loss: {running_loss / len(train_loader):.4f}")


# ---------------------------------------------------------------------------
# 4. Save checkpoint
# ---------------------------------------------------------------------------

ckpt_path = os.path.join(OUT_ROOT, "checkpoints", "model.pt")
torch.save(model.state_dict(), ckpt_path)
log.info(f"Checkpoint saved to {ckpt_path}")


# ---------------------------------------------------------------------------
# 5. Evaluate clean + shifted. Record dataset_idx for every sample so that
#    clean and shifted versions match by identity, not by position.
# ---------------------------------------------------------------------------

def predict_on_dataset(model, ds_clean, ds_shifted, device):
    """Return list of Sample objects indexed by dataset index.

    For each i we pull both clean[i] and shifted[i] to guarantee identity.
    """
    model.eval()
    samples: list[Sample] = []
    with torch.no_grad():
        for i in range(len(ds_clean)):
            clean_img, true_lbl = ds_clean[i]
            shifted_img, _ = ds_shifted[i]

            # Clean prediction
            c_out = model(clean_img.unsqueeze(0).to(device))
            c_probs = F.softmax(c_out, dim=1)
            c_conf, c_pred = torch.max(c_probs, 1)

            # Shifted prediction + confidence/entropy/margin
            s_out = model(shifted_img.unsqueeze(0).to(device))
            s_probs = F.softmax(s_out, dim=1)
            s_conf, s_pred = torch.max(s_probs, 1)
            top2 = s_probs.topk(2).values.squeeze(0)
            margin = float((top2[0] - top2[1]).item()) if len(top2) >= 2 else 0.0
            entropy = float(-torch.sum(s_probs * torch.log(s_probs + 1e-9)).item())

            samples.append(
                Sample(
                    index=i,
                    clean=clean_img.unsqueeze(0),
                    shifted=shifted_img.unsqueeze(0),
                    true_label=int(true_lbl),
                    pred_label_shifted=int(s_pred.item()),
                    confidence_shifted=float(s_conf.item()),
                    entropy_shifted=entropy,
                    margin_shifted=margin,
                    shifted_correct=(s_pred.item() == true_lbl),
                    clean_pred_label=int(c_pred.item()),
                    clean_confidence=float(c_conf.item()),
                )
            )
    return samples


all_samples = predict_on_dataset(model, test_dataset_clean, test_dataset_shift, device)

shifted_correct = [s for s in all_samples if s.shifted_correct]
shifted_incorrect = [s for s in all_samples if not s.shifted_correct]

clean_acc = sum(1 for s in all_samples if s.clean_pred_label == s.true_label) / len(all_samples)
shifted_acc = len(shifted_correct) / len(all_samples)

log.info(f"Clean accuracy  : {clean_acc:.4f}")
log.info(f"Shifted accuracy: {shifted_acc:.4f}")
log.info(f"Shifted incorrect (failures) : {len(shifted_incorrect)}")
log.info(f"Shifted correct   (survived) : {len(shifted_correct)}")


# ---------------------------------------------------------------------------
# 6. Reproducible subset selection (fixed seed, not first-N).
# ---------------------------------------------------------------------------

def reproducible_subset(samples: list[Sample], n: int, seed: int) -> list[Sample]:
    if len(samples) <= n:
        return list(samples)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), size=min(n, len(samples)), replace=False)
    idx.sort()
    return [samples[int(i)] for i in idx]


analysis_incorrect = reproducible_subset(shifted_incorrect, N_SAMPLES_PROFILE, seed=SEED)
analysis_correct = reproducible_subset(shifted_correct, N_SAMPLES_PROFILE, seed=SEED)

log.info(f"Profiling {len(analysis_incorrect)} shifted-incorrect, "
         f"{len(analysis_correct)} shifted-correct samples")


# ---------------------------------------------------------------------------
# 7. PNR calibration on clean-vs-clean (null) population, then profile.
# ---------------------------------------------------------------------------

log.info("Running Cascade...")
cascade = Cascade(model, device=str(device))
log.info(f"Instrumented layers: {cascade.layer_names}")

def build_null_pairs(ds_clean, n_pairs: int = 200):
    """Clean-vs-clean null pairs: same image paired with itself to estimate
    the minimum D(k) achievable under identical inputs."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(ds_clean), size=min(n_pairs, len(ds_clean)), replace=False)
    for i in idx:
        x, lbl = ds_clean[int(i)]
        yield (x.unsqueeze(0), x.unsqueeze(0), int(lbl))


null_pairs = list(build_null_pairs(test_dataset_clean, n_pairs=200))
pnr_thresholds = calibrate_pnr_thresholds(
    cascade, null_pairs, quantile=PNR_QUANTILE
)
log.info(
    f"PNR thresholds (clean-vs-clean q{PNR_QUANTILE:.2f}): "
    f"{[(n, f'{v:.4f}') for n, v in zip(pnr_thresholds.layer_names, pnr_thresholds.values)]}"
)

profile = fragility_profile(
    cascade,
    samples_incorrect=analysis_incorrect,
    samples_correct=analysis_correct,
)
print()
print(profile.summary())


# ---------------------------------------------------------------------------
# 8. Per-sample diagnosis on first 3 shifted-incorrect + 3 shifted-correct
# ---------------------------------------------------------------------------

print("\n--- Per-sample diagnosis (shifted-incorrect, first 3) ---")
for s in analysis_incorrect[:3]:
    r: DiagnosisResult = diagnose(
        cascade, s.clean, s.shifted, s.true_label, s.pred_label_shifted,
        threshold=pnr_thresholds,
    )
    print(f"\nSample idx={s.index} true={s.true_label} pred={s.pred_label_shifted} "
          f"(conf={s.confidence_shifted:.3f}, entropy={s.entropy_shifted:.3f})")
    print(r.summary())

print("\n--- Per-sample diagnosis (shifted-correct, first 3) ---")
for s in analysis_correct[:3]:
    r: DiagnosisResult = diagnose(
        cascade, s.clean, s.shifted, s.true_label, s.pred_label_shifted,
        threshold=pnr_thresholds,
    )
    print(f"\nSample idx={s.index} true={s.true_label} pred={s.pred_label_shifted} "
          f"(conf={s.confidence_shifted:.3f}, entropy={s.entropy_shifted:.3f})")
    print(r.summary())


# ---------------------------------------------------------------------------
# 9. Save artifacts
# ---------------------------------------------------------------------------

env_snapshot = {
    "python_version": sys.version,
    "platform": platform.platform(),
    "torch_version": torch.__version__,
    "torchvision_version": getattr(torchvision, "__version__", "unknown"),
    "numpy_version": np.__version__,
    "device": str(device),
}

config = {
    "experiment": EXPERIMENT_NAME,
    "seed": SEED,
    "device": str(device),
    "shift": {
        "name": SHIFT_NAME,
        **asdict(SHIFT_SPEC),
    },
    "training": {
        "epochs": N_TRAIN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "adam",
        "lr": 0.001,
        "loss": "cross_entropy",
    },
    "profile": {
        "n_samples_per_group": N_SAMPLES_PROFILE,
        "pnr_quantile": PNR_QUANTILE,
        "pnr_layers": pnr_thresholds.layer_names,
        "pnr_values": list(pnr_thresholds.values),
    },
    "accuracy": {
        "clean": clean_acc,
        "shifted": shifted_acc,
        "n_shifted_correct": len(shifted_correct),
        "n_shifted_incorrect": len(shifted_incorrect),
    },
    "layers": cascade.layer_names,
}

with open(os.path.join(OUT_ROOT, "config.json"), "w") as f:
    json.dump(config, f, indent=2, default=str)

with open(os.path.join(OUT_ROOT, "environment.json"), "w") as f:
    json.dump(env_snapshot, f, indent=2)

metrics_path = os.path.join(OUT_ROOT, "metrics.json")
_ = save_summary_json(profile, metrics_path)
log.info(f"Metrics written to {metrics_path}")

plots_dir = os.path.join(OUT_ROOT, "plots")
plot_files = save_layer_profiles(profile, plots_dir)
log.info(f"Plots/CSVs saved: {plot_files}")

# predictions.csv (full test set)
pred_path = os.path.join(OUT_ROOT, "predictions.csv")
with open(pred_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "index", "true_label", "clean_pred", "clean_conf",
        "shifted_pred", "shifted_conf", "entropy", "margin",
        "shifted_correct",
    ])
    for s in all_samples:
        w.writerow([
            s.index, s.true_label, s.clean_pred_label, s.clean_confidence,
            s.pred_label_shifted, s.confidence_shifted, s.entropy_shifted,
            s.margin_shifted, 1 if s.shifted_correct else 0,
        ])
log.info(f"Full predictions saved to {pred_path}")

# sample_metadata.csv (subset analyzed)
meta_path = os.path.join(OUT_ROOT, "sample_metadata.csv")
with open(meta_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["index", "group", "true_label", "pred_label", "confidence", "entropy", "margin"])
    for s in analysis_incorrect:
        w.writerow([s.index, "shifted_incorrect", s.true_label,
                    s.pred_label_shifted, s.confidence_shifted,
                    s.entropy_shifted, s.margin_shifted])
    for s in analysis_correct:
        w.writerow([s.index, "shifted_correct", s.true_label,
                    s.pred_label_shifted, s.confidence_shifted,
                    s.entropy_shifted, s.margin_shifted])
log.info(f"Analyzed-sample metadata saved to {meta_path}")

log.info(f"Done. All artifacts under {OUT_ROOT}")
