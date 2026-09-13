#!/usr/bin/env python3
"""
Experiment 3 — PNR statistical layer on ResNet18 / CIFAR-10.

Does NOT import train_and_Run_resnet18.py (that script trains on import).
Architecture, CIFAR normalize-after-shift, MAX_LAYERS=8, and the runner's
mild ShiftSpec are copied here so this is the same model family as the
16× amplification writeup — not a rerun of the MNIST toy CNN.

Split (all disjoint, sized before launch):
  calib    θ_k (naive 95th + Bonferroni)
  tune     Option C multiplier, template frozen from calib (refit; do not
           reuse the MNIST m≈2.02)
  holdout  union FAR + Wilson CI — the number to cite
  mwu_null / mwu_shifted  MWU+BH only; not used for θ_k, m, or FAR

Default n=300 per fold: DKW is wider than the n=500 PDF example, but 1500
ResNet18 GradCAM pairs is a session we can finish. Do not start n=500 and
time out.

Writes results/pnr_statistical_layer_resnet18.json — report whatever comes
out; do not backfill a 5% target.
"""

from __future__ import annotations

import argparse
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
import torchvision
import torchvision.models as models
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
from cascade.pnr import PNRThresholds
from cascade.shift import ShiftSpec, build_shift
from cascade.significance import layer_significance_from_trajectories

# Match train_and_Run_resnet18.py
SEED = 42
MAX_LAYERS = 8
SHIFT_SPEC = ShiftSpec(degrees=45.0, blur_kernel=5, blur_sigma=1.5)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
NULL_NOISE_SIGMA = 0.05  # pixel-space two-view, then CIFAR-normalize

# Session budget (override with flags). 300×5 = 1500 GradCAM pairs.
N_CALIB = 300
N_TUNE = 300
N_HOLDOUT = 300
N_MWU = 300
N_TRAIN_EPOCHS = 5
BATCH_SIZE = 128


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_cifar_resnet18() -> nn.Module:
    """Same CIFAR stem as train_and_Run_resnet18.py / exp01."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


class NormWrapper(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
        self._to_tensor = T.ToTensor()
        self._norm = T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if not isinstance(x, torch.Tensor):
            x = self._to_tensor(x)
        return self._norm(x), y


class ShiftedNormDataset(torch.utils.data.Dataset):
    """Deterministic shift on [0, 1], then CIFAR normalize."""

    def __init__(self, base, spec: ShiftSpec):
        self.base = base
        self._shift = build_shift(spec)
        self._to_tensor = T.ToTensor()
        self._norm = T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if not isinstance(x, torch.Tensor):
            x = self._to_tensor(x)
        return self._norm(self._shift(x)), y


def _train(model, loader, epochs, device, log):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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


def _chw01(x: torch.Tensor) -> torch.Tensor:
    t = x.detach().cpu()
    if t.dim() == 4:
        t = t.squeeze(0)
    return t


def _norm_batch1(x_chw: torch.Tensor) -> torch.Tensor:
    return T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)(x_chw).unsqueeze(0)


def _pairs_two_view_raw(raw_ds, indices, sigma, seed0):
    """Two-view null in pixel space, then the same CIFAR normalize as the model."""
    to_tensor = T.ToTensor()
    pairs = []
    for j, i in enumerate(indices):
        x, lbl = raw_ds[int(i)]
        if not isinstance(x, torch.Tensor):
            x = to_tensor(x)
        x = _chw01(x)
        a, b = gaussian_two_view(
            x, sigma, seed_a=seed0 + 2 * j, seed_b=seed0 + 2 * j + 1, clamp01=True
        )
        pairs.append((_norm_batch1(_chw01(a)), _norm_batch1(_chw01(b)), int(lbl)))
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


def _collect_logged(cascade, pairs, tag, log) -> np.ndarray:
    rows = []
    t0 = time.time()
    n = len(pairs)
    for i, (a, b, y) in enumerate(pairs):
        _, dk_norm = cascade.dk(a, b, target_class=y)
        rows.append([float(v) for v in dk_norm])
        done = i + 1
        if done == 10 or done % 50 == 0 or done == n:
            elapsed = time.time() - t0
            rate = elapsed / done
            log.info(
                "%s %d/%d  %.2fs/pair  ETA %.0fs",
                tag, done, n, rate, rate * (n - done),
            )
    return np.asarray(rows, dtype=float)


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
    lo, hi = far.wilson_ci()
    return {
        "n_triggered": far.n_triggered,
        "n": far.n,
        "rate": far.rate,
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


def _layer_means(traj: np.ndarray, names):
    return [
        {
            "layer_name": names[k],
            "mean": float(np.mean(traj[:, k])),
            "median": float(np.median(traj[:, k])),
        }
        for k in range(traj.shape[1])
    ]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-calib", type=int, default=N_CALIB)
    p.add_argument("--n-tune", type=int, default=N_TUNE)
    p.add_argument("--n-holdout", type=int, default=N_HOLDOUT)
    p.add_argument("--n-mwu", type=int, default=N_MWU)
    p.add_argument("--epochs", type=int, default=N_TRAIN_EPOCHS)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Load this state_dict instead of training.")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def run(args):
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = _device()
    out_root = os.path.join(
        _REPO_ROOT,
        "results",
        f"pnr_statistical_layer_resnet18_seed{seed}_{int(time.time())}",
    )
    os.makedirs(out_root, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(out_root, "run.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("exp03")
    n_calib, n_tune, n_hold, n_mwu = (
        args.n_calib, args.n_tune, args.n_holdout, args.n_mwu,
    )
    n_dk = n_calib + n_tune + n_hold + n_mwu + n_mwu
    log.info("device=%s out=%s", device, out_root)
    log.info(
        "budget: calib=%d tune=%d holdout=%d mwu_null=%d mwu_shifted=%d "
        "total_dk_pairs=%d  DKW ε(n_calib=%d,δ=0.05)=%.4f  ε_sim(L=%d)=%.4f",
        n_calib, n_tune, n_hold, n_mwu, n_mwu, n_dk, n_calib,
        dkw_epsilon(n_calib), MAX_LAYERS,
        dkw_epsilon_simultaneous(n_calib, MAX_LAYERS),
    )
    log.info("Option C multiplier will be fit on the ResNet18 tune fold; MNIST m is not used.")

    data_root = os.path.join(_REPO_ROOT, "data")
    train_raw = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=None
    )
    test_raw = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=None
    )
    train_ds = NormWrapper(train_raw)
    test_clean = NormWrapper(test_raw)
    test_shift = ShiftedNormDataset(test_raw, SHIFT_SPEC)

    model = build_cifar_resnet18().to(device)
    ckpt_path = os.path.join(out_root, "resnet18_cifar10.pt")
    if args.checkpoint:
        log.info("loading checkpoint %s", args.checkpoint)
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        model.eval()
        ckpt_path = args.checkpoint
    else:
        loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
        )
        log.info("training CIFAR ResNet18 for %d epochs", args.epochs)
        _train(model, loader, args.epochs, device, log)
        torch.save(model.state_dict(), ckpt_path)
        log.info("saved %s", ckpt_path)

    cascade = Cascade(model, max_layers=MAX_LAYERS, device=str(device))
    log.info("layers (%d): %s", cascade.n_layers, cascade.layer_names)
    if cascade.n_layers != MAX_LAYERS:
        raise RuntimeError(
            f"expected {MAX_LAYERS} layers, got {cascade.n_layers}: {cascade.layer_names}"
        )

    rng = np.random.default_rng(seed)
    all_idx = rng.choice(len(test_clean), size=n_dk, replace=False)
    i0 = 0

    def take(n):
        nonlocal i0
        sl = all_idx[i0 : i0 + n]
        i0 += n
        return sl

    calib_idx = take(n_calib)
    tune_idx = take(n_tune)
    hold_idx = take(n_hold)
    mwu_null_idx = take(n_mwu)
    mwu_shift_idx = take(n_mwu)
    assert i0 == n_dk
    # Disjointness is guaranteed by sampling without replacement.
    folds = [calib_idx, tune_idx, hold_idx, mwu_null_idx, mwu_shift_idx]
    seen = np.concatenate(folds)
    assert len(np.unique(seen)) == n_dk

    null_calib = _pairs_two_view_raw(test_raw, calib_idx, NULL_NOISE_SIGMA, 1000)
    null_tune = _pairs_two_view_raw(test_raw, tune_idx, NULL_NOISE_SIGMA, 5000)
    null_hold = _pairs_two_view_raw(test_raw, hold_idx, NULL_NOISE_SIGMA, 9000)
    mwu_null = _pairs_two_view_raw(test_raw, mwu_null_idx, NULL_NOISE_SIGMA, 13000)
    mwu_shifted = _pairs_clean_shifted(test_clean, test_shift, mwu_shift_idx)

    log.info("collecting D(k) trajectories")
    calib_traj = _collect_logged(cascade, null_calib, "calib", log)
    tune_traj = _collect_logged(cascade, null_tune, "tune", log)
    hold_traj = _collect_logged(cascade, null_hold, "holdout", log)
    mwu_null_traj = _collect_logged(cascade, mwu_null, "mwu_null", log)
    mwu_shift_traj = _collect_logged(cascade, mwu_shifted, "mwu_shifted", log)

    L = cascade.n_layers
    naive_vals = thresholds_at_quantile(calib_traj, 0.95)
    bonf_q = 1.0 - 0.05 / L
    bonf_vals = thresholds_at_quantile(calib_traj, bonf_q)

    joint_vals, joint_m, joint_tune_far = calibrate_joint_from_trajectories(
        tune_traj, target_fdr=0.05, base_thresholds=naive_vals
    )
    tune_far_after = trajectory_false_alarm_rate(tune_traj, joint_vals)
    option_c_notes = []
    if joint_m <= 0 or not np.isfinite(joint_m):
        option_c_notes.append("multiplier is non-finite or non-positive")
    if joint_tune_far > 0.05 + 1e-9:
        option_c_notes.append(
            f"tune-fold FAR {joint_tune_far:.4f} exceeds target 0.05"
        )
    if tune_far_after.rate == 0.0:
        option_c_notes.append(
            "tune-fold FAR is 0 — search may have jumped over the 5% target"
        )
    if joint_m > 10:
        option_c_notes.append(f"multiplier {joint_m:.3f} is large vs MNIST-scale ~2")

    naive = _make_thresholds(
        cascade.layer_names, naive_vals, 0.95, n_calib, "naive",
        trajectory_false_alarm_rate(calib_traj, naive_vals).rate,
    )
    bonf = _make_thresholds(
        cascade.layer_names, bonf_vals, bonf_q, n_calib, "bonferroni",
        trajectory_false_alarm_rate(calib_traj, bonf_vals).rate,
    )
    joint = _make_thresholds(
        cascade.layer_names, joint_vals, 0.95, n_calib, "joint",
        joint_tune_far,
    )

    hold_far = {
        "naive": _far_record(trajectory_false_alarm_rate(hold_traj, naive.values)),
        "bonferroni": _far_record(trajectory_false_alarm_rate(hold_traj, bonf.values)),
        "joint": _far_record(trajectory_false_alarm_rate(hold_traj, joint.values)),
    }
    for name, rec in hold_far.items():
        log.info("holdout FAR %s: %s", name, rec["display"])
    log.info("Option C multiplier (ResNet18 tune fold)=%.4f notes=%s", joint_m, option_c_notes)

    log.info("layer significance MWU + BH on dedicated folds (not calib/tune/holdout)")
    table = layer_significance_from_trajectories(
        cascade.layer_names, mwu_null_traj, mwu_shift_traj, q=0.05, fdr_method="bh"
    )
    table_by = layer_significance_from_trajectories(
        cascade.layer_names, mwu_null_traj, mwu_shift_traj, q=0.05, fdr_method="by"
    )

    payload = {
        "experiment": "pnr_statistical_layer_resnet18",
        "dataset": "CIFAR-10",
        "model": "cifar10_resnet18",
        "seed": seed,
        "shift": {"name": "runner_mild", **asdict(SHIFT_SPEC)},
        "null": {
            "kind": "gaussian_two_view_then_cifar_normalize",
            "sigma_pixel": NULL_NOISE_SIGMA,
            "note": (
                "Two-view noise is applied in [0,1] pixel space, then the same "
                "CIFAR mean/std as train_and_Run_resnet18.py. Identical-image "
                "pairs are not used (degenerate D(k))."
            ),
        },
        "n_calib": n_calib,
        "n_tune": n_tune,
        "n_holdout": n_hold,
        "n_mwu_null": n_mwu,
        "n_mwu_shifted": n_mwu,
        "split": {
            "calib": "per-layer θ_k only",
            "tune": "Option C multiplier; naive template frozen from calib",
            "holdout": "union FAR + Wilson 95% CI",
            "mwu_null": "MWU null group; disjoint from calib/tune/holdout",
            "mwu_shifted": "MWU shifted group; disjoint from all null folds",
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
            "tune_joint": _far_record(tune_far_after),
            "holdout_wilson_95": hold_far,
            "option_c_multiplier": joint_m,
            "option_c_notes": option_c_notes,
            "note": (
                "Cite holdout_wilson_95[*].display. Multiplier was refit on this "
                "ResNet18 tune fold; do not carry over the MNIST value."
            ),
        },
        "descriptive_d_norm": {
            "note": (
                "Mean/median D_norm on the MWU folds only — descriptive, not a "
                "causal compounding test (Section 7 not run)."
            ),
            "mwu_null": _layer_means(mwu_null_traj, cascade.layer_names),
            "mwu_shifted": _layer_means(mwu_shift_traj, cascade.layer_names),
        },
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
        "checkpoint": ckpt_path,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
    }

    json_path = os.path.join(out_root, "pnr_statistical_layer.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("wrote %s", json_path)

    csv_path = os.path.join(out_root, "layer_significance.csv")
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

    latest = os.path.join(_REPO_ROOT, "results", "pnr_statistical_layer_resnet18.json")
    with open(latest, "w") as f:
        json.dump(payload, f, indent=2)
    latest_csv = os.path.join(_REPO_ROOT, "results", "layer_significance_resnet18.csv")
    with open(csv_path, "r") as src, open(latest_csv, "w") as dst:
        dst.write(src.read())
    log.info("also wrote %s and %s", latest, latest_csv)
    return payload


if __name__ == "__main__":
    run(parse_args())
