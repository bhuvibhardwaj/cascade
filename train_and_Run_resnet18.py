"""
ResNet18 on CIFAR-10 — hardened version.

Same guarantees as train_and_run.py:
  1. Dataset-indexed clean/shifted pairing
  2. Genuinely deterministic shift (functional rotate + fixed sigma)
  3. Structured Samples with index + prediction metadata
  4. Full reproducibility (Python / NumPy / PyTorch seeds, recorded)
  5. Checkpoint + artifacts: config, environment, metrics, CSVs, plots, log
  6. Four-group analysis: shifted-correct + shifted-incorrect
  7. Reproducible subset sampling (fixed RNG seed)

Run: python3 train_and_Run_resnet18.py
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
import torchvision.models as models
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
EXPERIMENT_NAME = "cifar10_resnet18"
SHIFT_NAME = "mild"
SHIFT_SPEC = ShiftSpec(degrees=45.0, blur_kernel=5, blur_sigma=1.5)
N_TRAIN_EPOCHS = 5
BATCH_SIZE = 128
N_SAMPLES_PROFILE = 100
PNR_QUANTILE = 0.95
MAX_LAYERS = 8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

OUT_ROOT = os.path.join(
    "results", f"{EXPERIMENT_NAME}_{SHIFT_SPEC.label()}_seed{SEED}_{int(time.time())}"
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
log = logging.getLogger("resnet18")
log.info(f"Device: {device}")
log.info(f"Output root: {OUT_ROOT}")
log.info(f"Shift spec: {asdict(SHIFT_SPEC)}")

# ---------------------------------------------------------------------------
# 1. Data — clean + deterministic-shifted (CIFAR-10 normalized)
# ---------------------------------------------------------------------------

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
transform_norm = T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
to_tensor = T.ToTensor()

# CIFAR-10 channel normalization is applied AFTER the shift (so the shifted
# image is a transformed version of the same underlying normalized sample).
# Build shifted dataset from PIL base, then apply norm to both.

def _compose_norm(tensor_img: torch.Tensor) -> torch.Tensor:
    return transform_norm(tensor_img)


train_dataset_raw = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=None
)
test_dataset_raw = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=None
)

class NormWrapper(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        if not isinstance(x, torch.Tensor):
            x = to_tensor(x)
        return _compose_norm(x), y

class ShiftedNormDataset(torch.utils.data.Dataset):
    """Apply deterministic shift THEN normalize."""
    def __init__(self, base, spec: ShiftSpec):
        self.base = base
        self._shift = build_shift(spec)
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        if not isinstance(x, torch.Tensor):
            x = to_tensor(x)
        return _compose_norm(self._shift(x)), y


train_dataset = NormWrapper(train_dataset_raw)
test_dataset_clean = NormWrapper(test_dataset_raw)
test_dataset_shift = ShiftedNormDataset(test_dataset_raw, SHIFT_SPEC)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
    num_workers=0,
)
test_loader_clean = torch.utils.data.DataLoader(
    test_dataset_clean, batch_size=BATCH_SIZE, shuffle=False
)
test_loader_shift = torch.utils.data.DataLoader(
    test_dataset_shift, batch_size=BATCH_SIZE, shuffle=False
)


# ---------------------------------------------------------------------------
# 2. Model — ResNet18 adapted for 32x32 CIFAR-10
# ---------------------------------------------------------------------------

model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(512, 10)
model = model.to(device)


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
# 4. Checkpoint
# ---------------------------------------------------------------------------

ckpt_path = os.path.join(OUT_ROOT, "checkpoints", "model.pt")
torch.save(model.state_dict(), ckpt_path)
log.info(f"Checkpoint saved to {ckpt_path}")


# ---------------------------------------------------------------------------
# 5. Accuracy evaluation (batched) + full indexed per-sample predictions
# ---------------------------------------------------------------------------

def evaluate_accuracy(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


clean_acc = evaluate_accuracy(test_loader_clean)
shifted_acc = evaluate_accuracy(test_loader_shift)
log.info(f"Clean accuracy  : {clean_acc:.2f}%")
log.info(f"Shifted accuracy: {shifted_acc:.2f}%")


def predict_on_dataset(model, ds_clean, ds_shifted, device):
    model.eval()
    samples: list[Sample] = []
    with torch.no_grad():
        for i in range(len(ds_clean)):
            clean_img, true_lbl = ds_clean[i]
            shifted_img, _ = ds_shifted[i]
            c_out = model(clean_img.unsqueeze(0).to(device))
            c_probs = F.softmax(c_out, dim=1)
            c_conf, c_pred = torch.max(c_probs, 1)
            s_out = model(shifted_img.unsqueeze(0).to(device))
            s_probs = F.softmax(s_out, dim=1)
            s_conf, s_pred = torch.max(s_probs, 1)
            top2 = s_probs.topk(min(2, s_probs.shape[1])).values.squeeze(0)
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


log.info("Building full indexed predictions (test set)...")
all_samples = predict_on_dataset(model, test_dataset_clean, test_dataset_shift, device)

shifted_correct = [s for s in all_samples if s.shifted_correct]
shifted_incorrect = [s for s in all_samples if not s.shifted_correct]
log.info(f"Shifted incorrect (failures) : {len(shifted_incorrect)}")
log.info(f"Shifted correct   (survived) : {len(shifted_correct)}")


# ---------------------------------------------------------------------------
# 6. Reproducible subset sampling
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
# 7. Cascade profile + PNR calibration
# ---------------------------------------------------------------------------

log.info("Running Cascade...")
cascade = Cascade(model, max_layers=MAX_LAYERS, device=str(device))
log.info(f"Instrumented layers: {cascade.layer_names}")


def build_null_pairs(ds_clean, n_pairs: int = 200):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(ds_clean), size=min(n_pairs, len(ds_clean)), replace=False)
    for i in idx:
        x, lbl = ds_clean[int(i)]
        yield (x.unsqueeze(0), x.unsqueeze(0), int(lbl))


null_pairs = list(build_null_pairs(test_dataset_clean, n_pairs=200))
pnr_thresholds = calibrate_pnr_thresholds(cascade, null_pairs, quantile=PNR_QUANTILE)
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
# 8. Per-sample diagnosis
# ---------------------------------------------------------------------------

print("\n--- Per-sample diagnosis (shifted-incorrect, first 5) ---")
for s in analysis_incorrect[:5]:
    r: DiagnosisResult = diagnose(
        cascade, s.clean, s.shifted, s.true_label, s.pred_label_shifted,
        threshold=pnr_thresholds,
    )
    print(f"\nSample idx={s.index} true={s.true_label} pred={s.pred_label_shifted} "
          f"(conf={s.confidence_shifted:.3f}, entropy={s.entropy_shifted:.3f})")
    print(r.summary())

print("\n--- Per-sample diagnosis (shifted-correct, first 5) ---")
for s in analysis_correct[:5]:
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
        "label": SHIFT_SPEC.label(),
        **asdict(SHIFT_SPEC),
    },
    "training": {
        "epochs": N_TRAIN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "adam",
        "lr": 1e-3,
        "loss": "cross_entropy",
        "cifar_mean": CIFAR_MEAN,
        "cifar_std": CIFAR_STD,
    },
    "profile": {
        "n_samples_per_group": N_SAMPLES_PROFILE,
        "pnr_quantile": PNR_QUANTILE,
        "pnr_layers": pnr_thresholds.layer_names,
        "pnr_values": list(pnr_thresholds.values),
        "max_layers": MAX_LAYERS,
    },
    "accuracy_pct": {
        "clean": clean_acc,
        "shifted": shifted_acc,
    },
    "n_groups": {
        "shifted_correct": len(shifted_correct),
        "shifted_incorrect": len(shifted_incorrect),
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
