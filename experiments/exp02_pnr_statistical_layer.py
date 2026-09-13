#!/usr/bin/env python3
"""
Experiment 2 — statistical layer for population-calibrated PNR thresholds.

Does NOT import train_and_run.py / train_and_Run_resnet18.py (those train on
import). Reuses the same MNIST CNN family, deterministic ShiftSpec presets,
and clean/shifted pairing-by-index as those scripts.

Three-way null split: calib (θ_k) / tune (Option C multiplier) / holdout
(union FAR with Wilson CI). This MNIST run is the cheap check that the split
behaves; ResNet18/CIFAR-10 is the scheduled GPU job for numbers to cite.

Writes measured numbers under results/pnr_statistical_layer_* — no placeholders.
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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from cascade import Cascade
from cascade.bounds import (
    calibrate_joint_from_trajectories,
    dkw_epsilon,
    dkw_epsilon_simultaneous,
    thresholds_at_quantile,
    trajectory_false_alarm_rate,
)
from cascade.metrics import gaussian_two_view
from cascade.pnr import PNRThresholds, collect_dk_trajectories
from cascade.shift import get_preset_spec, make_shifted_dataset
from cascade.significance import layer_significance_table


SEED = 42
SHIFT_NAME = "aggressive"
N_TRAIN_EPOCHS = 2
# Three disjoint null folds + a shifted MWU sample. 500 calib matches the
# math-backing DKW worked example; 350 holdout is in the ±10pp Wilson-width
# regime around a 5% rate (300+ for a tighter ±5pp interval is the next step).
N_CALIB = 500
N_TUNE = 350
N_HOLDOUT = 350
N_SHIFTED = 350
NULL_NOISE_SIGMA = 0.05  # two-view benign null (math-backing §2: seed/augmentation)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
OUT_ROOT = os.path.join(
    _REPO_ROOT,
    "results",
    f"pnr_statistical_layer_mnist_seed{SEED}_{int(time.time())}",
)
os.makedirs(OUT_ROOT, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUT_ROOT, "run.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("exp02")
log.info("device=%s out=%s", device, OUT_ROOT)


class EightLayerCNN(nn.Module):
    """Eight Conv2d layers so L matches the math-backing ResNet18/CIFAR writeup."""

    def __init__(self):
        super().__init__()
        chans = [1, 16, 16, 32, 32, 32, 48, 48, 64]
        blocks = []
        for i in range(8):
            blocks.append(nn.Conv2d(chans[i], chans[i + 1], 3, padding=1))
            blocks.append(nn.ReLU(inplace=True))
            if i in (1, 3, 5):
                blocks.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(x)


class MNISTCNN(nn.Module):
    """Same architecture as train_and_run.py (2 convs)."""

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


def _train(model, loader, epochs, device):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            loss = crit(model(images), labels)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        log.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, total / max(n, 1))
    model.eval()


def _pairs_two_view(ds, indices, sigma, seed0):
    pairs = []
    for j, i in enumerate(indices):
        x, lbl = ds[int(i)]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        a, b = gaussian_two_view(
            x, sigma, seed_a=seed0 + 2 * j, seed_b=seed0 + 2 * j + 1, clamp01=True
        )
        pairs.append((a, b, int(lbl)))
    return pairs


def _pairs_clean_shifted(ds_clean, ds_shift, indices):
    pairs = []
    for i in indices:
        x, lbl = ds_clean[int(i)]
        y, _ = ds_shift[int(i)]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if y.dim() == 3:
            y = y.unsqueeze(0)
        pairs.append((x, y, int(lbl)))
    return pairs


def _thresholds_record(obj):
    return {
        "layer_names": obj.layer_names,
        "values": list(obj.values),
        "quantile": obj.quantile,
        "n_pairs": obj.n_pairs,
        "epsilon": obj.epsilon,
        "epsilon_simultaneous": obj.epsilon_simultaneous,
        "calibration_method": obj.calibration_method,
        "measured_false_alarm_rate_on_fit_fold": obj.measured_false_alarm_rate,
    }


def _far_record(far):
    rate, lo, hi = far.rate, *far.wilson_ci()
    return {
        "n_triggered": far.n_triggered,
        "n": far.n,
        "rate": rate,
        "wilson_95_lo": lo,
        "wilson_95_hi": hi,
        "display": far.format_ci(),
    }


def _make_thresholds(layer_names, values, quantile, n_pairs, method, fit_far_rate):
    L = len(layer_names)
    return PNRThresholds(
        layer_names=list(layer_names),
        values=[float(v) for v in values],
        quantile=float(quantile),
        n_pairs=int(n_pairs),
        epsilon=dkw_epsilon(n_pairs),
        epsilon_simultaneous=dkw_epsilon_simultaneous(n_pairs, L),
        calibration_method=method,
        measured_false_alarm_rate=float(fit_far_rate),
    )


def run():
    spec = get_preset_spec(SHIFT_NAME)
    clean_tf = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.MNIST(
        root=os.path.join(_REPO_ROOT, "data"), train=True, download=True, transform=clean_tf
    )
    test_raw = torchvision.datasets.MNIST(
        root=os.path.join(_REPO_ROOT, "data"), train=False, download=True, transform=None
    )
    test_clean = torchvision.datasets.MNIST(
        root=os.path.join(_REPO_ROOT, "data"), train=False, download=True, transform=clean_tf
    )
    test_shift = make_shifted_dataset(test_raw, spec, to_tensor=True)

    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    # Primary: L=8 on real MNIST (matches the 8-layer writeup structure).
    model = EightLayerCNN().to(device)
    log.info("training EightLayerCNN")
    _train(model, loader, N_TRAIN_EPOCHS, device)
    ckpt = os.path.join(OUT_ROOT, "eight_layer_cnn.pt")
    torch.save(model.state_dict(), ckpt)

    cascade = Cascade(model, device=str(device))
    log.info("layers (%d): %s", cascade.n_layers, cascade.layer_names)
    assert cascade.n_layers == 8

    rng = np.random.default_rng(SEED)
    n_needed = N_CALIB + N_TUNE + N_HOLDOUT + N_SHIFTED
    all_idx = rng.choice(len(test_clean), size=n_needed, replace=False)
    i0 = 0
    calib_idx = all_idx[i0 : i0 + N_CALIB]; i0 += N_CALIB
    tune_idx = all_idx[i0 : i0 + N_TUNE]; i0 += N_TUNE
    hold_idx = all_idx[i0 : i0 + N_HOLDOUT]; i0 += N_HOLDOUT
    shift_idx = all_idx[i0 : i0 + N_SHIFTED]

    null_calib = _pairs_two_view(test_clean, calib_idx, NULL_NOISE_SIGMA, seed0=1000)
    null_tune = _pairs_two_view(test_clean, tune_idx, NULL_NOISE_SIGMA, seed0=5000)
    null_hold = _pairs_two_view(test_clean, hold_idx, NULL_NOISE_SIGMA, seed0=9000)
    shifted = _pairs_clean_shifted(test_clean, test_shift, shift_idx)

    log.info(
        "collecting D(k): calib=%d tune=%d holdout=%d shifted=%d",
        len(null_calib), len(null_tune), len(null_hold), len(shifted),
    )
    calib_traj = collect_dk_trajectories(cascade, null_calib)
    tune_traj = collect_dk_trajectories(cascade, null_tune)
    hold_traj = collect_dk_trajectories(cascade, null_hold)

    L = cascade.n_layers
    naive_vals = thresholds_at_quantile(calib_traj, 0.95)
    bonf_q = 1.0 - 0.05 / L
    bonf_vals = thresholds_at_quantile(calib_traj, bonf_q)
    joint_vals, joint_m, joint_tune_far = calibrate_joint_from_trajectories(
        tune_traj, target_fdr=0.05, base_thresholds=naive_vals
    )

    naive = _make_thresholds(
        cascade.layer_names, naive_vals, 0.95, N_CALIB, "naive",
        trajectory_false_alarm_rate(calib_traj, naive_vals).rate,
    )
    bonf = _make_thresholds(
        cascade.layer_names, bonf_vals, bonf_q, N_CALIB, "bonferroni",
        trajectory_false_alarm_rate(calib_traj, bonf_vals).rate,
    )
    joint = _make_thresholds(
        cascade.layer_names, joint_vals, 0.95, N_CALIB, "joint",
        joint_tune_far,
    )

    hold_far = {
        "naive": _far_record(trajectory_false_alarm_rate(hold_traj, naive.values)),
        "bonferroni": _far_record(trajectory_false_alarm_rate(hold_traj, bonf.values)),
        "joint": _far_record(trajectory_false_alarm_rate(hold_traj, joint.values)),
    }
    for name, rec in hold_far.items():
        log.info("holdout FAR %s: %s", name, rec["display"])
    log.info("Option C multiplier (tuned on disjoint fold)=%.4f", joint_m)

    log.info("layer significance MWU + BH")
    table = layer_significance_table(
        cascade, null_hold, shifted, q=0.05, fdr_method="bh"
    )
    table_by = layer_significance_table(
        cascade, null_hold, shifted, q=0.05, fdr_method="by"
    )

    payload = {
        "experiment": "pnr_statistical_layer",
        "dataset": "MNIST",
        "model": "EightLayerCNN",
        "seed": SEED,
        "shift": {"name": SHIFT_NAME, **asdict(spec)},
        "null": {
            "kind": "gaussian_two_view",
            "sigma": NULL_NOISE_SIGMA,
            "note": (
                "Math-backing §2: null pairs from the same unshifted distribution, "
                "differing by seed/augmentation. Identical-image pairs are degenerate "
                "(D(k)≈0) and are not used here."
            ),
        },
        "n_calib": N_CALIB,
        "n_tune": N_TUNE,
        "n_holdout": N_HOLDOUT,
        "n_shifted": N_SHIFTED,
        "split": {
            "calib": "per-layer θ_k (naive 95th and Bonferroni 1-α/L)",
            "tune": "Option C multiplier; template frozen from calib",
            "holdout": "final union FAR with Wilson 95% CI — the number to cite",
        },
        "n_layers": cascade.n_layers,
        "layer_names": cascade.layer_names,
        "dkw": {
            "n_pairs": naive.n_pairs,
            "delta": 0.05,
            "epsilon": dkw_epsilon(naive.n_pairs),
            "epsilon_simultaneous": dkw_epsilon_simultaneous(
                naive.n_pairs, cascade.n_layers
            ),
        },
        "thresholds": {
            "naive": _thresholds_record(naive),
            "bonferroni": _thresholds_record(bonf),
            "joint": _thresholds_record(joint),
        },
        "false_alarm_rate": {
            "independence_formula_L8_q95": 1.0 - (0.95**8),
            "holdout_wilson_95": hold_far,
            "note": (
                "Cite holdout_wilson_95[*].display, not a bare percentage. "
                "ResNet18/CIFAR-10 is the setup to cite for the 16× amplification "
                "finding; this MNIST run is the cheap three-way-split check."
            ),
        },
        "joint_multiplier": joint_m,
        "layer_significance_bh": {
            "q": table.q,
            "fdr_method": table.fdr_method,
            "n_null": table.n_null,
            "n_shifted": table.n_shifted,
            "cutoff_rank": table.cutoff_rank,
            "rows": table.as_records(),
        },
        "layer_significance_by": {
            "q": table_by.q,
            "fdr_method": table_by.fdr_method,
            "n_null": table_by.n_null,
            "n_shifted": table_by.n_shifted,
            "cutoff_rank": table_by.cutoff_rank,
            "rows": table_by.as_records(),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
    }

    json_path = os.path.join(OUT_ROOT, "pnr_statistical_layer.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("wrote %s", json_path)

    csv_path = os.path.join(OUT_ROOT, "layer_significance.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "layer_name",
                "layer_index",
                "u_statistic",
                "z_score",
                "p_value",
                "p_adjusted_bh",
                "rejected_bh",
                "p_adjusted_by",
                "rejected_by",
            ],
        )
        w.writeheader()
        for a, b in zip(table.rows, table_by.rows):
            w.writerow(
                {
                    "layer_name": a.layer_name,
                    "layer_index": a.layer_index,
                    "u_statistic": a.u_statistic,
                    "z_score": a.z_score,
                    "p_value": a.p_value,
                    "p_adjusted_bh": a.p_adjusted,
                    "rejected_bh": a.rejected,
                    "p_adjusted_by": b.p_adjusted,
                    "rejected_by": b.rejected,
                }
            )
    log.info("wrote %s", csv_path)

    # Stable pointer for RESULTS.md
    latest = os.path.join(_REPO_ROOT, "results", "pnr_statistical_layer.json")
    with open(latest, "w") as f:
        json.dump(payload, f, indent=2)
    latest_csv = os.path.join(_REPO_ROOT, "results", "layer_significance.csv")
    with open(csv_path, "r") as src, open(latest_csv, "w") as dst:
        dst.write(src.read())
    log.info("also wrote %s and %s", latest, latest_csv)
    return payload


if __name__ == "__main__":
    run()
