#!/usr/bin/env python3
"""
Experiment 1 — D(k) metric robustness.

Does NOT train. Does NOT import train_and_run.py / train_and_Run_resnet18.py
(those scripts train on import). Requires --checkpoint.

Scientific question:
    Does the observed increase of D(k) with depth survive when raw L2
    scale effects are controlled?

This run does not test causal propagation (H1). Z_identity, Z_noise, and
Z_augmentation are sensitivity analyses under different reference
distributions, not three confirmations of one hypothesis.

Identity is a numerical-noise diagnostic. Tiny Gaussian and mild
augmentation are benign-perturbation reference distributions, not canonical
null hypotheses.

---------------------------------------------------------------------------
Function signatures (Experiment 1 helpers live in cascade.metrics / core):

    Cascade.attribution_maps(image, target_class) -> List[Tensor]
    cosine_distance(a, b, eps=COSINE_EPSILON) -> float
    d_raw_d_rel_from_maps(a_clean, a_shift, eps=NORM_EPSILON) -> (D_raw, D_rel)
        D_rel uses the existing D_norm formula; dk() is not modified.
    layer_metrics_from_maps(maps_clean, maps_shift) -> (D_raw, D_rel, D_cos)
    gaussian_two_view(x, sigma, seed_a, seed_b, clamp01) -> (view_a, view_b)
    assess_null_degeneracy(values, std_min, maxabs_min) -> DegeneracyDecision
    standardize_z(d_shift, null_mean, null_std, decision, eps) -> ndarray
        NaN if decision.degenerate; never substitutes another reference.

---------------------------------------------------------------------------
Output schema (written under --out-dir):

    config.json                 seeds, indices, transforms, model/shift, degeneracy rules
    environment.json
    sample_metadata.csv         evaluation indices + prediction metadata
    null_calibration_indices.csv
    null_identity.csv           per calib sample × layer D_raw
    null_noise.csv
    null_augmentation.csv
    null_summary.json           mean/std/n + DegeneracyDecision per layer × reference
    per_sample_metrics.csv      eval D_raw, D_rel (=D_norm), D_cos, Z_* (NaN if degenerate)
    summary.json                layer-wise stats, growth, paired tests, correlations
    plots/metrics_vs_depth.png
    plots/metrics_overlay_layer0_normalized.png
    plots/metric_correlations_last_layer.png
    run.log
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import random
import sys
import time

# Prefer this repository's package over any other `cascade` on PYTHONPATH.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from scipy import stats

from cascade import Cascade, Sample, ShiftSpec, build_shift, get_preset_spec, make_shifted_dataset
from cascade.core import NORM_EPSILON
from cascade.metrics import (
    Z_EPSILON,
    apply_with_isolated_torch_seed,
    assess_null_degeneracy,
    gaussian_two_view,
    layer_metrics_from_maps,
    standardize_z,
)

SCIENTIFIC_SCOPE = (
    "Experiment 1 tests whether depth-growth of attribution drift is robust "
    "to metric scale. It does not provide causal evidence of upstream "
    "propagation. Identity is a numerical-noise diagnostic. Tiny Gaussian "
    "and mild augmentation are benign-perturbation reference distributions. "
    "Z_identity, Z_noise, and Z_augmentation are sensitivity analyses under "
    "those references, not independent confirmations of one hypothesis."
)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
NOISE_STD_DEFAULT = 1e-3
N_NULL_DEFAULT = 200
SEED_NULL_OFFSET = 1  # calibration RNG is seed + 1
VIEW_SEED_MULT = 1_000_003


# Architectures must match train_and_run.py / train_and_Run_resnet18.py.
# Copied here only so this script can load_state_dict without importing
# those runners (they train at import time).


class MNISTCNN(nn.Module):
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


def build_cifar_resnet18() -> nn.Module:
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
    """Deterministic shift on [0, 1], then CIFAR normalize — same as the ResNet runner."""

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


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not math.isfinite(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return obj


def require_checkpoint(path: str) -> str:
    if not path:
        raise SystemExit("ERROR: --checkpoint is required. Experiment 1 will not train.")
    if not os.path.isfile(path):
        raise SystemExit(
            f"ERROR: checkpoint does not exist: {path}\n"
            "Experiment 1 will not train a new model. Provide the original "
            "weights file (state_dict .pt) from a prior run."
        )
    return os.path.abspath(path)


def view_seeds(experiment_seed: int, dataset_index: int) -> Tuple[int, int]:
    base = int(experiment_seed) * VIEW_SEED_MULT + int(dataset_index) * 2
    return base, base + 1


def reproducible_subset(samples: List[Sample], n: int, seed: int) -> List[Sample]:
    if len(samples) <= n:
        return list(samples)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), size=n, replace=False)
    idx.sort()
    return [samples[int(i)] for i in idx]


def predict_on_dataset(model, ds_clean, ds_shifted, device) -> List[Sample]:
    model.eval()
    samples: List[Sample] = []
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


def mild_aug_transform(dataset: str):
    if dataset == "mnist":
        return T.Compose(
            [
                T.RandomCrop(28, padding=2),
                T.ToTensor(),
            ]
        )
    if dataset == "cifar10":
        return T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
            ]
        )
    raise ValueError(dataset)


def _summarise(layer_values: List[List[float]]) -> dict:
    means, stds, ci_lo, ci_hi, meds, ns = [], [], [], [], [], []
    for vals in layer_values:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        n = int(arr.size)
        ns.append(n)
        if n == 0:
            means.append(float("nan"))
            stds.append(float("nan"))
            ci_lo.append(float("nan"))
            ci_hi.append(float("nan"))
            meds.append(float("nan"))
            continue
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        med = float(np.median(arr))
        if n > 1:
            se = s / math.sqrt(n)
            t_crit = float(stats.t.ppf(0.975, df=n - 1))
            lo, hi = m - t_crit * se, m + t_crit * se
        else:
            lo = hi = m
        means.append(m)
        stds.append(s)
        ci_lo.append(lo)
        ci_hi.append(hi)
        meds.append(med)
    growth = []
    for i in range(1, len(means)):
        if means[i - 1] in (0.0, None) or not math.isfinite(means[i - 1]) or means[i - 1] == 0:
            growth.append(float("nan"))
        else:
            growth.append(float(means[i] / means[i - 1]))
    return {
        "n": ns,
        "mean": means,
        "std": stds,
        "median": meds,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "growth_ratio": growth,
    }


def _paired_tests(layer_values: List[List[float]], layer_names: List[str]) -> List[dict]:
    """Paired adjacent-layer tests. Primary: t if Shapiro p>=0.05 else Wilcoxon."""
    out: List[dict] = []
    for k in range(1, len(layer_values)):
        a = np.asarray(layer_values[k], dtype=float)
        b = np.asarray(layer_values[k - 1], dtype=float)
        n = min(len(a), len(b))
        mask = np.isfinite(a[:n]) & np.isfinite(b[:n])
        av, bv = a[:n][mask], b[:n][mask]
        rec = {
            "layer_prev": layer_names[k - 1],
            "layer": layer_names[k],
            "n_paired": int(mask.sum()),
            "shapiro_W": None,
            "shapiro_p": None,
            "paired_t_stat": None,
            "paired_t_p": None,
            "wilcoxon_stat": None,
            "wilcoxon_p": None,
            "primary_test": None,
            "primary_p": None,
            "note": None,
        }
        if av.size < 3:
            rec["note"] = "insufficient_paired_finite_samples"
            rec["primary_test"] = "none"
            out.append(rec)
            continue
        diffs = av - bv
        if np.allclose(diffs, 0.0):
            rec["note"] = "all_paired_differences_zero"
            rec["primary_test"] = "none"
            out.append(rec)
            continue
        try:
            w_s, p_s = stats.shapiro(diffs)
            rec["shapiro_W"] = float(w_s)
            rec["shapiro_p"] = float(p_s)
            shapiro_ok = float(p_s) >= 0.05
        except ValueError:
            rec["note"] = "shapiro_failed"
            shapiro_ok = False
        try:
            t_stat, t_p = stats.ttest_rel(av, bv)
            rec["paired_t_stat"] = float(t_stat)
            rec["paired_t_p"] = float(t_p)
        except ValueError:
            rec["note"] = (rec["note"] + ";" if rec["note"] else "") + "ttest_rel_failed"
        try:
            w_res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            rec["wilcoxon_stat"] = float(w_res.statistic)
            rec["wilcoxon_p"] = float(w_res.pvalue)
        except ValueError:
            rec["note"] = (rec["note"] + ";" if rec["note"] else "") + "wilcoxon_failed"
        if shapiro_ok and rec["paired_t_p"] is not None:
            rec["primary_test"] = "paired_t"
            rec["primary_p"] = rec["paired_t_p"]
        elif rec["wilcoxon_p"] is not None:
            rec["primary_test"] = "wilcoxon_signed_rank"
            rec["primary_p"] = rec["wilcoxon_p"]
        else:
            rec["primary_test"] = "none"
        out.append(rec)
    return out


def _corr_pair(x, y) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(x.size), "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {
        "n": int(x.size),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


def _write_null_csv(path: str, layer_names: List[str], indices: Sequence[int], matrix: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "layer_index", "layer_name", "d_raw"])
        for row_i, ds_i in enumerate(indices):
            for k, name in enumerate(layer_names):
                w.writerow([int(ds_i), k, name, float(matrix[row_i, k])])


def save_plots(
    out_dir: str,
    layer_names: List[str],
    groups: Dict[str, Dict[str, List[List[float]]]],
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    produced: List[str] = []
    xs = np.arange(len(layer_names))
    colors = {"shifted_incorrect": "#c0392b", "shifted_correct": "#27ae60"}
    metric_keys = [
        ("d_raw", "D_raw (existing L2)"),
        ("d_rel", "D_rel = existing D_norm"),
        ("d_cos", "D_cos"),
        ("z_noise", "Z_noise (benign Gaussian reference)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (key, title) in zip(axes.ravel(), metric_keys):
        for gname, gmetrics in groups.items():
            arr = np.asarray(gmetrics[key], dtype=float)  # n x K
            if arr.size == 0:
                continue
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
            ax.plot(xs, mean, marker="o", label=gname, color=colors.get(gname, "gray"))
            ax.fill_between(xs, mean - std, mean + std, color=colors.get(gname, "gray"), alpha=0.15)
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel("mean ± SD")
    axes[1, 0].set_ylabel("mean ± SD")
    fig.suptitle("Experiment 1: metric vs depth (not a causal test)")
    fig.tight_layout()
    p = os.path.join(out_dir, "metrics_vs_depth.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    produced.append(p)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    overlay = [
        ("d_raw", "D_raw"),
        ("d_rel", "D_rel (D_norm)"),
        ("d_cos", "D_cos"),
        ("z_noise", "Z_noise"),
        ("z_augmentation", "Z_augmentation"),
        ("z_identity", "Z_identity"),
    ]
    # Overlay uses shifted-incorrect means, layer-0 normalized for visual scale.
    g = groups.get("shifted_incorrect") or next(iter(groups.values()), None)
    if g is not None:
        for key, lab in overlay:
            arr = np.asarray(g[key], dtype=float)
            mean = np.nanmean(arr, axis=0)
            if not np.isfinite(mean[0]) or mean[0] == 0:
                continue
            ax.plot(xs, mean / mean[0], marker="o", label=lab)
        ax.axhline(1.0, color="gray", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean(k) / mean(layer 0)")
    ax.set_title("Scale-free overlay (failed group). Z omitted if degenerate (NaN).")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "metrics_overlay_layer0_normalized.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    produced.append(p)

    names = ["d_raw", "d_rel", "d_cos"]
    fig, axes = plt.subplots(1, max(1, len(groups)), figsize=(5 * max(1, len(groups)), 4))
    if len(groups) == 1:
        axes = [axes]
    for ax, (gname, gmetrics) in zip(axes, groups.items()):
        mats = [np.asarray(gmetrics[k], dtype=float) for k in names]
        last = np.column_stack([m[:, -1] for m in mats])
        corr = np.full((3, 3), np.nan)
        for i in range(3):
            for j in range(3):
                r = _corr_pair(last[:, i], last[:, j])["pearson_r"]
                corr[i, j] = np.nan if r is None else r
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        ax.set_title(f"Pearson last-layer ({gname})")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p = os.path.join(out_dir, "metric_correlations_last_layer.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    produced.append(p)
    return produced


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Experiment 1: D(k) metric robustness (no training)")
    p.add_argument("--checkpoint", required=True, help="Path to existing state_dict. Run fails if missing.")
    p.add_argument("--dataset", required=True, choices=["mnist", "cifar10"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--n-eval-per-group", type=int, default=None)
    p.add_argument("--n-null", type=int, default=N_NULL_DEFAULT)
    p.add_argument("--noise-std", type=float, default=NOISE_STD_DEFAULT)
    p.add_argument("--max-layers", type=int, default=None, help="Override; CIFAR default is 8 to match the existing runner.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ckpt = require_checkpoint(args.checkpoint)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    dataset = args.dataset
    n_eval = args.n_eval_per_group
    max_layers = args.max_layers
    if dataset == "mnist":
        if n_eval is None:
            n_eval = 200
        if max_layers is None:
            max_layers = None
        shift_name = "aggressive"
        shift_spec = get_preset_spec(shift_name)
        model_name = "mnist_cnn"
    else:
        if n_eval is None:
            n_eval = 100
        if max_layers is None:
            max_layers = 8
        shift_name = "custom_45_blur5_sigma1.5"  # existing runner's SHIFT_NAME is 'mild' but spec is not PRESETS['mild']
        shift_spec = ShiftSpec(degrees=45.0, blur_kernel=5, blur_sigma=1.5)
        model_name = "cifar10_resnet18"

    stamp = int(time.time())
    out_dir = args.out_dir or os.path.join(
        "results",
        "exp01_dk_metric_robustness",
        f"{dataset}_{model_name}_{shift_spec.label()}_seed{seed}_{stamp}",
    )
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(out_dir, "run.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("exp01")
    log.info(SCIENTIFIC_SCOPE)
    log.info(f"checkpoint={ckpt}")
    log.info(f"device={device}  out_dir={out_dir}")

    if dataset == "mnist":
        model = MNISTCNN()
        to_tensor = T.ToTensor()
        ds_raw = torchvision.datasets.MNIST(
            root=args.data_root, train=False, download=True, transform=None
        )
        ds_clean = torchvision.datasets.MNIST(
            root=args.data_root, train=False, download=True, transform=to_tensor
        )
        ds_shift = make_shifted_dataset(ds_raw, shift_spec, to_tensor=True)
        clamp01 = True
        aug_desc = {"ops": ["RandomCrop(28, padding=2)", "ToTensor"], "horizontal_flip": False}
    else:
        model = build_cifar_resnet18()
        ds_raw = torchvision.datasets.CIFAR10(
            root=args.data_root, train=False, download=True, transform=None
        )
        ds_clean = NormWrapper(ds_raw)
        ds_shift = ShiftedNormDataset(ds_raw, shift_spec)
        clamp01 = False
        aug_desc = {
            "ops": [
                "RandomCrop(32, padding=4)",
                "RandomHorizontalFlip(p=0.5)",
                "ToTensor",
                "Normalize(CIFAR mean/std)",
            ],
            "horizontal_flip": True,
            "cifar_mean": list(CIFAR_MEAN),
            "cifar_std": list(CIFAR_STD),
            "note": "Flip is omitted on MNIST because it can change digit identity.",
        }

    state = torch.load(ckpt, map_location=str(device))
    if isinstance(state, nn.Module):
        raise SystemExit(
            "ERROR: checkpoint is a full nn.Module, expected a state_dict "
            f"as saved by the existing runners ({ckpt})."
        )
    if isinstance(state, dict) and "state_dict" in state and not any(
        k.startswith("conv") or k.startswith("fc") or k.startswith("layer") for k in state
    ):
        state = state["state_dict"]
    try:
        model.load_state_dict(state)
    except Exception as exc:
        raise SystemExit(
            f"ERROR: failed to load state_dict into {model_name}: {exc}\n"
            "Architecture is reconstructed to match the existing experiment "
            "runners. This script will not train."
        ) from exc
    model = model.to(device).eval()

    log.info("Evaluating full test set (for group membership only)...")
    all_samples = predict_on_dataset(model, ds_clean, ds_shift, device)
    shifted_correct = [s for s in all_samples if s.shifted_correct]
    shifted_incorrect = [s for s in all_samples if not s.shifted_correct]
    clean_correct_n = sum(1 for s in all_samples if s.clean_pred_label == s.true_label)
    log.info(
        f"test n={len(all_samples)} clean_correct={clean_correct_n} "
        f"shifted_correct={len(shifted_correct)} shifted_incorrect={len(shifted_incorrect)}"
    )

    analysis_incorrect = reproducible_subset(shifted_incorrect, n_eval, seed=seed)
    analysis_correct = reproducible_subset(shifted_correct, n_eval, seed=seed)
    eval_samples = analysis_incorrect + analysis_correct
    eval_indices = sorted({s.index for s in eval_samples})
    remaining = [i for i in range(len(ds_clean)) if i not in set(eval_indices)]
    if not remaining:
        raise SystemExit("ERROR: no disjoint test indices remain for reference-distribution calibration.")
    rng_null = np.random.default_rng(seed + SEED_NULL_OFFSET)
    n_null = min(int(args.n_null), len(remaining))
    calib_indices = rng_null.choice(remaining, size=n_null, replace=False)
    calib_indices = np.sort(calib_indices).astype(int)
    overlap = set(eval_indices).intersection(set(calib_indices.tolist()))
    if overlap:
        raise SystemExit(f"ERROR: calibration/eval index overlap: {sorted(overlap)[:20]}")

    log.info(
        f"eval incorrect={len(analysis_incorrect)} correct={len(analysis_correct)} "
        f"calib_null={len(calib_indices)} (test holdout; not train)"
    )

    cascade = Cascade(model, max_layers=max_layers, device=str(device))
    layer_names = list(cascade.layer_names)
    n_layers = cascade.n_layers
    log.info(f"layers={layer_names}")

    # --- Reference distributions on disjoint test calib indices ---
    aug_tf = mild_aug_transform(dataset)
    noise_std = float(args.noise_std)
    refs = {
        "identity": np.zeros((n_null, n_layers), dtype=float),
        "noise": np.zeros((n_null, n_layers), dtype=float),
        "augmentation": np.zeros((n_null, n_layers), dtype=float),
    }
    log.info("Computing identity / Gaussian / mild-augmentation reference D_raw...")
    for row, i in enumerate(calib_indices.tolist()):
        x_clean, y = ds_clean[int(i)]
        pil, y_raw = ds_raw[int(i)]
        label = int(y)
        assert int(y_raw) == label
        d_id, _ = cascade.dk(x_clean, x_clean, target_class=label)
        seed_a, seed_b = view_seeds(seed, int(i))
        va, vb = gaussian_two_view(x_clean, noise_std, seed_a, seed_b, clamp01=clamp01)
        d_n, _ = cascade.dk(va, vb, target_class=label)
        aa = apply_with_isolated_torch_seed(seed_a, lambda: aug_tf(pil))
        ab = apply_with_isolated_torch_seed(seed_b, lambda: aug_tf(pil))
        d_a, _ = cascade.dk(aa, ab, target_class=label)
        refs["identity"][row] = d_id
        refs["noise"][row] = d_n
        refs["augmentation"][row] = d_a
        if (row + 1) % 25 == 0:
            log.info(f"  reference pairs {row + 1}/{n_null}")

    null_summary = {"n_calibration": n_null, "references": {}}
    z_plans = {}
    for ref_name, mat in refs.items():
        per_layer = []
        means, stds = [], []
        decisions = []
        for k in range(n_layers):
            col = mat[:, k]
            dec = assess_null_degeneracy(col)
            decisions.append(dec)
            means.append(float(np.mean(col)))
            stds.append(float(np.std(col, ddof=1)) if n_null > 1 else 0.0)
            per_layer.append(
                {
                    "layer": layer_names[k],
                    "mean_d_raw": means[-1],
                    "std_d_raw": stds[-1],
                    "n": n_null,
                    "degeneracy": dec.to_dict(),
                }
            )
            if dec.degenerate:
                log.info(
                    f"  {ref_name} layer {layer_names[k]} DEGENERATE ({dec.reason}); "
                    f"Z will be NaN. No substitute reference."
                )
        role = {
            "identity": "numerical_noise_diagnostic",
            "noise": "benign_perturbation_reference_tiny_gaussian",
            "augmentation": "benign_perturbation_reference_mild_aug",
        }[ref_name]
        null_summary["references"][ref_name] = {
            "role": role,
            "layers": per_layer,
        }
        z_plans[ref_name] = {"mean": means, "std": stds, "decision": decisions}

    # --- Evaluation metrics on shift pairs ---
    log.info("Computing evaluation D_raw / D_rel / D_cos...")
    rows_out = []
    group_store: Dict[str, Dict[str, List[List[float]]]] = {
        "shifted_incorrect": {k: [] for k in ["d_raw", "d_rel", "d_cos", "z_identity", "z_noise", "z_augmentation"]},
        "shifted_correct": {k: [] for k in ["d_raw", "d_rel", "d_cos", "z_identity", "z_noise", "z_augmentation"]},
    }
    checked_dk = False
    for s in eval_samples:
        maps_c = cascade.attribution_maps(s.clean, s.true_label)
        maps_s = cascade.attribution_maps(s.shifted, s.true_label)
        d_raw, d_rel, d_cos = layer_metrics_from_maps(maps_c, maps_s)
        if not checked_dk:
            dk_raw, dk_norm = cascade.dk(s.clean, s.shifted, s.true_label)
            if any(abs(a - b) > 1e-5 for a, b in zip(d_raw, dk_raw)) or any(
                abs(a - b) > 1e-5 for a, b in zip(d_rel, dk_norm)
            ):
                log.warning(
                    "D_raw/D_rel from maps differ from a second dk() pass "
                    f"(expected small GradCAM rerun noise). maps raw={d_raw} dk={dk_raw}"
                )
            checked_dk = True
        z_id = [
            float(
                standardize_z(
                    [d_raw[k]],
                    z_plans["identity"]["mean"][k],
                    z_plans["identity"]["std"][k],
                    z_plans["identity"]["decision"][k],
                )[0]
            )
            for k in range(n_layers)
        ]
        z_n = [
            float(
                standardize_z(
                    [d_raw[k]],
                    z_plans["noise"]["mean"][k],
                    z_plans["noise"]["std"][k],
                    z_plans["noise"]["decision"][k],
                )[0]
            )
            for k in range(n_layers)
        ]
        z_a = [
            float(
                standardize_z(
                    [d_raw[k]],
                    z_plans["augmentation"]["mean"][k],
                    z_plans["augmentation"]["std"][k],
                    z_plans["augmentation"]["decision"][k],
                )[0]
            )
            for k in range(n_layers)
        ]
        grp = "shifted_incorrect" if not s.shifted_correct else "shifted_correct"
        group_store[grp]["d_raw"].append(d_raw)
        group_store[grp]["d_rel"].append(d_rel)
        group_store[grp]["d_cos"].append(d_cos)
        group_store[grp]["z_identity"].append(z_id)
        group_store[grp]["z_noise"].append(z_n)
        group_store[grp]["z_augmentation"].append(z_a)
        for k, name in enumerate(layer_names):
            rows_out.append(
                {
                    "index": s.index,
                    "group": grp,
                    "true_label": s.true_label,
                    "pred_label": s.pred_label_shifted,
                    "layer_index": k,
                    "layer_name": name,
                    "d_raw": d_raw[k],
                    "d_rel": d_rel[k],
                    "d_cos": d_cos[k],
                    "z_identity": z_id[k],
                    "z_noise": z_n[k],
                    "z_augmentation": z_a[k],
                }
            )

    # drop empty groups from plots
    group_store = {g: m for g, m in group_store.items() if m["d_raw"]}

    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "device": str(device),
    }
    config = {
        "experiment": "exp01_dk_metric_robustness",
        "scientific_scope": SCIENTIFIC_SCOPE,
        "seed": seed,
        "seed_eval_subset": seed,
        "seed_null_calibration": seed + SEED_NULL_OFFSET,
        "view_seed_formula": "experiment_seed * 1000003 + dataset_index * 2 (+1 for view B)",
        "device": str(device),
        "checkpoint": ckpt,
        "dataset": dataset,
        "model": model_name,
        "shift": {"reported_name": shift_name, **asdict(shift_spec)},
        "layers": layer_names,
        "max_layers": max_layers,
        "n_eval_per_group_requested": n_eval,
        "n_eval_shifted_incorrect": len(analysis_incorrect),
        "n_eval_shifted_correct": len(analysis_correct),
        "eval_indices": eval_indices,
        "n_null_requested": int(args.n_null),
        "n_null_used": n_null,
        "calib_indices": calib_indices.tolist(),
        "calib_eval_disjoint": True,
        "calib_source": "held_out_test_indices_not_in_eval",
        "calib_source_rationale": (
            "Reference μ/σ are estimated on test images that are not in the "
            "evaluation subset so evaluation trajectories are not used to fit "
            "the Z scale. Train is not used: it is a different image "
            "distribution and is unnecessary for a same-domain reference."
        ),
        "identity": {
            "role": "numerical_noise_diagnostic",
            "procedure": "D_raw(clean, clean) via existing Cascade.dk",
        },
        "noise": {
            "role": "benign_perturbation_reference",
            "procedure": "two independent N(0, sigma^2) views of the model-input tensor",
            "sigma": noise_std,
            "clamp01": clamp01,
            "applied_on": "model_input_tensor (MNIST [0,1]; CIFAR after mean/std normalize)",
        },
        "augmentation": {
            "role": "benign_perturbation_reference",
            "procedure": "two independently seeded mild augs; not rotation+blur shift",
            **aug_desc,
        },
        "degeneracy": {
            "std_min": 1e-8,
            "maxabs_min": 1e-6,
            "policy": "write NaN for that Z layer; do not substitute another reference",
        },
        "d_rel": "alias of existing D_norm from Cascade.dk arithmetic",
        "z_epsilon": Z_EPSILON,
        "norm_epsilon": NORM_EPSILON,
        "test_counts": {
            "n_test": len(all_samples),
            "n_clean_correct": clean_correct_n,
            "n_shifted_correct": len(shifted_correct),
            "n_shifted_incorrect": len(shifted_incorrect),
        },
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(json_safe(config), f, indent=2)
    with open(os.path.join(out_dir, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)
    with open(os.path.join(out_dir, "null_summary.json"), "w") as f:
        json.dump(json_safe(null_summary), f, indent=2)

    _write_null_csv(
        os.path.join(out_dir, "null_identity.csv"), layer_names, calib_indices, refs["identity"]
    )
    _write_null_csv(os.path.join(out_dir, "null_noise.csv"), layer_names, calib_indices, refs["noise"])
    _write_null_csv(
        os.path.join(out_dir, "null_augmentation.csv"),
        layer_names,
        calib_indices,
        refs["augmentation"],
    )
    with open(os.path.join(out_dir, "null_calibration_indices.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index"])
        for i in calib_indices.tolist():
            w.writerow([int(i)])

    meta_path = os.path.join(out_dir, "sample_metadata.csv")
    with open(meta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "index",
                "group",
                "true_label",
                "pred_label",
                "clean_pred",
                "clean_conf",
                "confidence_shifted",
                "entropy",
                "margin",
            ]
        )
        for s in eval_samples:
            w.writerow(
                [
                    s.index,
                    "shifted_incorrect" if not s.shifted_correct else "shifted_correct",
                    s.true_label,
                    s.pred_label_shifted,
                    s.clean_pred_label,
                    s.clean_confidence,
                    s.confidence_shifted,
                    s.entropy_shifted,
                    s.margin_shifted,
                ]
            )

    metrics_path = os.path.join(out_dir, "per_sample_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        fields = [
            "index",
            "group",
            "true_label",
            "pred_label",
            "layer_index",
            "layer_name",
            "d_raw",
            "d_rel",
            "d_cos",
            "z_identity",
            "z_noise",
            "z_augmentation",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows_out:
            out_row = dict(row)
            for zkey in ("z_identity", "z_noise", "z_augmentation"):
                if isinstance(out_row[zkey], float) and not math.isfinite(out_row[zkey]):
                    out_row[zkey] = "nan"
            w.writerow(out_row)

    metric_names = ["d_raw", "d_rel", "d_cos", "z_identity", "z_noise", "z_augmentation"]
    summary: dict = {
        "scientific_scope": SCIENTIFIC_SCOPE,
        "layer_names": layer_names,
        "groups": {},
        "adjacent_layer_tests": "paired_t if Shapiro-Wilk p>=0.05 on paired differences else Wilcoxon signed-rank; both reported",
    }
    for gname, gmetrics in group_store.items():
        gsum: dict = {"n_samples": len(gmetrics["d_raw"]), "metrics": {}}
        for mname in metric_names:
            cols = [[] for _ in range(n_layers)]
            for sample_row in gmetrics[mname]:
                for k in range(n_layers):
                    cols[k].append(sample_row[k])
            gsum["metrics"][mname] = {
                "summary": _summarise(cols),
                "paired_adjacent": _paired_tests(cols, layer_names),
            }
        # correlations per layer among d_raw, d_rel, d_cos
        corrs = {}
        for k, lname in enumerate(layer_names):
            raw = [row[k] for row in gmetrics["d_raw"]]
            rel = [row[k] for row in gmetrics["d_rel"]]
            cos = [row[k] for row in gmetrics["d_cos"]]
            corrs[lname] = {
                "d_raw_vs_d_rel": _corr_pair(raw, rel),
                "d_raw_vs_d_cos": _corr_pair(raw, cos),
                "d_rel_vs_d_cos": _corr_pair(rel, cos),
            }
        gsum["correlations"] = corrs
        summary["groups"][gname] = gsum

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(json_safe(summary), f, indent=2)

    plots = save_plots(os.path.join(out_dir, "plots"), layer_names, group_store)
    log.info(f"plots={plots}")
    log.info(f"Done. Artifacts in {out_dir}")
    print(out_dir)


if __name__ == "__main__":
    main()
