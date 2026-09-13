#!/usr/bin/env python3
"""
Experiment 4 — MVE: does frozen joint PNR predict shifted prediction failure?

Freezes the Exp 03 ResNet18/CIFAR-10 checkpoint and joint θ. Does not retrain
and does not recalibrate thresholds on shifted data.

Default run is the minimum viable experiment: n_eval=200 test indices disjoint
from the reconstructed Exp 03 folds. All rows are written (including
clean-wrong). Confirmatory filtering is clean-correct only, after collection.

Abort if clean-correct failures < 20. Do not retune the shift or θ.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
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
import torchvision.models as models
import torchvision.transforms as T

from cascade import Cascade
from cascade.shift import ShiftSpec, build_shift

SEED = 42
MAX_LAYERS = 8
N_EVAL = 200
MIN_FAILURES = 20
N_BOOT = 2000
CIFAR_N_TEST = 10000
SHIFT_SPEC = ShiftSpec(degrees=45.0, blur_kernel=5, blur_sigma=1.5)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

DEFAULT_EXP03_JSON = os.path.join(
    _REPO_ROOT, "results", "pnr_statistical_layer_resnet18.json"
)
DEFAULT_CHECKPOINT = os.path.join(
    _REPO_ROOT,
    "results",
    "pnr_statistical_layer_resnet18_seed42_1789106507",
    "resnet18_cifar10.pt",
)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def reconstruct_exp03_indices(exp03: dict) -> np.ndarray:
    """Replay Exp 03's unused-index draw. Exp 03 did not persist fold indices."""
    seed = int(exp03["seed"])
    n_dk = (
        int(exp03["n_calib"])
        + int(exp03["n_tune"])
        + int(exp03["n_holdout"])
        + int(exp03["n_mwu_null"])
        + int(exp03["n_mwu_shifted"])
    )
    rng = np.random.default_rng(seed)
    return rng.choice(CIFAR_N_TEST, size=n_dk, replace=False)


def pnr_index(d_norm, theta) -> int:
    for k, (d, t) in enumerate(zip(d_norm, theta)):
        if d > t:
            return k
    return len(theta)


def _softmax_stats(logits: torch.Tensor):
    probs = F.softmax(logits, dim=1).squeeze(0)
    conf, pred = torch.max(probs, dim=0)
    top2 = torch.topk(probs, k=min(2, probs.numel())).values
    margin = float((top2[0] - top2[1]).item()) if top2.numel() >= 2 else 0.0
    entropy = float((-torch.sum(probs * torch.log(probs + 1e-9))).item())
    return int(pred.item()), float(conf.item()), entropy, margin, probs.detach().cpu().numpy()


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average ranks for ties.
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_pos = float(ranks[y].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bootstrap_auc(y: np.ndarray, scores: np.ndarray, n_boot: int, rng: np.random.Generator):
    point = roc_auc(y, scores)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = roc_auc(y[idx], scores[idx])
        if np.isfinite(val):
            boots.append(val)
    if not boots:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def bootstrap_delta_auc(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
):
    point = roc_auc(y, score_a) - roc_auc(y, score_b)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        va = roc_auc(y[idx], score_a[idx])
        vb = roc_auc(y[idx], score_b[idx])
        if np.isfinite(va) and np.isfinite(vb):
            boots.append(va - vb)
    if not boots:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--exp03-json", type=str, default=DEFAULT_EXP03_JSON)
    p.add_argument("--n-eval", type=int, default=N_EVAL)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--min-failures", type=int, default=MIN_FAILURES)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    return p.parse_args()


def run(args):
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Exp 03 checkpoint not found: {args.checkpoint}. Do not retrain."
        )
    if not os.path.isfile(args.exp03_json):
        raise FileNotFoundError(f"Exp 03 results JSON not found: {args.exp03_json}")

    with open(args.exp03_json) as f:
        exp03 = json.load(f)

    joint = exp03["thresholds"]["joint"]
    theta = [float(v) for v in joint["values"]]
    if len(theta) != MAX_LAYERS:
        raise RuntimeError(f"expected {MAX_LAYERS} joint thresholds, got {len(theta)}")

    exp03_indices = reconstruct_exp03_indices(exp03)
    exp03_set = set(int(i) for i in exp03_indices.tolist())
    remaining = [i for i in range(CIFAR_N_TEST) if i not in exp03_set]
    if len(remaining) < args.n_eval:
        raise RuntimeError(
            f"only {len(remaining)} unused test indices; need n_eval={args.n_eval}"
        )

    rng = np.random.default_rng(seed)
    eval_indices = rng.choice(remaining, size=int(args.n_eval), replace=False)
    eval_indices = np.asarray(eval_indices, dtype=int)

    device = _device()
    out_root = os.path.join(
        _REPO_ROOT,
        "results",
        f"exp04_pnr_predictive_resnet18_seed{seed}_{int(time.time())}",
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
    log = logging.getLogger("exp04")
    log.info("device=%s out=%s", device, out_root)
    log.info("checkpoint=%s (frozen; not retraining)", args.checkpoint)
    log.info("joint θ frozen from %s: %s", args.exp03_json, theta)
    log.info("reconstructed exp03 n=%d; remaining=%d; n_eval=%d",
             len(exp03_indices), len(remaining), args.n_eval)

    splits = {
        "note": (
            "Exp 03 did not save fold indices. exp03_indices are reconstructed "
            "by replaying np.random.default_rng(seed).choice(10000, n_dk, "
            "replace=False) with seed and fold sizes from the Exp 03 JSON."
        ),
        "exp03_seed": int(exp03["seed"]),
        "exp03_n_dk": int(len(exp03_indices)),
        "exp03_indices": [int(i) for i in exp03_indices.tolist()],
        "eval_seed": seed,
        "eval_indices": [int(i) for i in eval_indices.tolist()],
    }
    splits_path = os.path.join(out_root, "splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    log.info("wrote %s immediately after drawing eval indices", splits_path)

    thresholds_path = os.path.join(out_root, "thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(
            {
                "source": os.path.abspath(args.exp03_json),
                "method": "joint",
                "frozen": True,
                "layer_names": list(joint["layer_names"]),
                "values": theta,
                "option_c_multiplier": exp03.get("false_alarm_rate", {}).get(
                    "option_c_multiplier"
                ),
            },
            f,
            indent=2,
        )

    data_root = os.path.join(_REPO_ROOT, "data")
    test_raw = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=None
    )
    test_clean = NormWrapper(test_raw)
    shift_fn = build_shift(SHIFT_SPEC)
    to_tensor = T.ToTensor()
    norm = T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)

    model = build_cifar_resnet18().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    cascade = Cascade(model, max_layers=MAX_LAYERS, device=str(device))
    log.info("layers (%d): %s", cascade.n_layers, cascade.layer_names)
    if cascade.n_layers != MAX_LAYERS:
        raise RuntimeError(
            f"expected {MAX_LAYERS} layers, got {cascade.n_layers}: {cascade.layer_names}"
        )
    if list(cascade.layer_names) != list(joint["layer_names"]):
        raise RuntimeError(
            f"layer names differ from Exp 03 joint θ: {cascade.layer_names} vs "
            f"{joint['layer_names']}"
        )

    fieldnames = [
        "dataset_index",
        "true_label",
        "pred_clean",
        "pred_shift",
        "correct_clean",
        "correct_shift",
        "failed",
        "conf_clean",
        "conf_shift",
        "entropy_shift",
        "margin_shift",
        *[f"d_norm_{k}" for k in range(MAX_LAYERS)],
        *[f"d_raw_{k}" for k in range(MAX_LAYERS)],
        "pnr_index",
        "pnr_triggered",
        "score_pnr",
        "max_d",
        "argmax_d",
        "d_last",
        "d_early_mean",
    ]

    rows = []
    t0 = time.time()
    n = len(eval_indices)
    for j, idx in enumerate(eval_indices):
        x_raw, y = test_raw[int(idx)]
        if not isinstance(x_raw, torch.Tensor):
            x01 = to_tensor(x_raw)
        else:
            x01 = x_raw
        x, y_chk = test_clean[int(idx)]
        y = int(y_chk)
        xs01 = shift_fn(x01)
        xs = norm(xs01)

        x_batch = x.unsqueeze(0) if x.dim() == 3 else x
        xs_batch = xs.unsqueeze(0) if xs.dim() == 3 else xs
        x_batch = x_batch.to(device)
        xs_batch = xs_batch.to(device)

        with torch.no_grad():
            clean_logits = model(x_batch)
            shift_logits = model(xs_batch)
        pred_clean, conf_clean, _, _, _ = _softmax_stats(clean_logits)
        pred_shift, conf_shift, entropy_shift, margin_shift, _ = _softmax_stats(
            shift_logits
        )

        d_raw, d_norm = cascade.dk(x_batch, xs_batch, target_class=y)
        d_raw = [float(v) for v in d_raw]
        d_norm = [float(v) for v in d_norm]
        pnr = pnr_index(d_norm, theta)
        correct_clean = pred_clean == y
        correct_shift = pred_shift == y
        failed = bool(correct_clean and (not correct_shift))
        row = {
            "dataset_index": int(idx),
            "true_label": y,
            "pred_clean": pred_clean,
            "pred_shift": pred_shift,
            "correct_clean": bool(correct_clean),
            "correct_shift": bool(correct_shift),
            "failed": failed,
            "conf_clean": conf_clean,
            "conf_shift": conf_shift,
            "entropy_shift": entropy_shift,
            "margin_shift": margin_shift,
            "pnr_index": int(pnr),
            "pnr_triggered": bool(pnr < MAX_LAYERS),
            "score_pnr": float(-pnr),
            "max_d": float(max(d_norm)),
            "argmax_d": int(int(np.argmax(d_norm))),
            "d_last": float(d_norm[-1]),
            "d_early_mean": float(np.mean(d_norm[:2])),
        }
        for k in range(MAX_LAYERS):
            row[f"d_norm_{k}"] = d_norm[k]
            row[f"d_raw_{k}"] = d_raw[k]
        rows.append(row)

        done = j + 1
        if done == 10 or done % 50 == 0 or done == n:
            elapsed = time.time() - t0
            rate = elapsed / done
            log.info(
                "eval %d/%d  %.2fs/pair  ETA %.0fs",
                done, n, rate, rate * (n - done),
            )

    csv_path = os.path.join(out_root, "samples.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s (%d rows, unfiltered)", csv_path, len(rows))

    n_raw = len(rows)
    eval_rows = [r for r in rows if r["correct_clean"]]
    n_cc = len(eval_rows)
    n_fail = sum(1 for r in eval_rows if r["failed"])
    fail_rate = (n_fail / n_cc) if n_cc else float("nan")

    log.info("raw: %d", n_raw)
    log.info("clean-correct: %d", n_cc)
    log.info("failures: %d", n_fail)
    log.info("failure rate: %s", f"{fail_rate:.6f}" if n_cc else "nan")

    aborted = n_fail < int(args.min_failures)
    analysis = {
        "experiment": "exp04_pnr_predictive_validity",
        "mode": "mve",
        "seed": seed,
        "checkpoint": os.path.abspath(args.checkpoint),
        "exp03_json": os.path.abspath(args.exp03_json),
        "shift": {"name": "runner_mild", **asdict(SHIFT_SPEC)},
        "n_layers": MAX_LAYERS,
        "layer_names": list(cascade.layer_names),
        "theta_method": "joint",
        "theta": theta,
        "n_eval_raw": n_raw,
        "n_clean_correct": n_cc,
        "n_failures": n_fail,
        "failure_rate": fail_rate,
        "min_failures": int(args.min_failures),
        "aborted": aborted,
        "abort_reason": (
            f"clean-correct failures {n_fail} < {args.min_failures}; "
            "insufficient for predictive inference. Shift and θ were not changed."
            if aborted
            else None
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
    }

    if aborted:
        log.warning("%s", analysis["abort_reason"])
    else:
        y = np.array([1 if r["failed"] else 0 for r in eval_rows], dtype=int)
        scores = {
            "score_pnr": np.array([r["score_pnr"] for r in eval_rows], dtype=float),
            "neg_conf_shift": np.array(
                [-r["conf_shift"] for r in eval_rows], dtype=float
            ),
            "d_last": np.array([r["d_last"] for r in eval_rows], dtype=float),
            "max_d": np.array([r["max_d"] for r in eval_rows], dtype=float),
            "d_early_mean": np.array(
                [r["d_early_mean"] for r in eval_rows], dtype=float
            ),
            "binary_pnr": np.array(
                [1.0 if r["pnr_triggered"] else 0.0 for r in eval_rows], dtype=float
            ),
        }
        boot_rng = np.random.default_rng(seed + 1)
        auc_table = {}
        for name, sc in scores.items():
            point, lo, hi = bootstrap_auc(y, sc, int(args.n_boot), boot_rng)
            auc_table[name] = {
                "auc": point,
                "ci95_lo": lo,
                "ci95_hi": hi,
            }
            log.info("AUC %s: %.4f [%.4f, %.4f]", name, point, lo, hi)

        delta_table = {}
        for other in ("neg_conf_shift", "d_last", "max_d", "d_early_mean"):
            d, lo, hi = bootstrap_delta_auc(
                y, scores["score_pnr"], scores[other], int(args.n_boot), boot_rng
            )
            delta_table[f"score_pnr_minus_{other}"] = {
                "delta_auc": d,
                "ci95_lo": lo,
                "ci95_hi": hi,
            }
            log.info("ΔAUC score_pnr - %s: %.4f [%.4f, %.4f]", other, d, lo, hi)

        trigger_all = float(np.mean([r["pnr_triggered"] for r in eval_rows]))
        failed_rows = [r for r in eval_rows if r["failed"]]
        surv_rows = [r for r in eval_rows if not r["failed"]]
        trigger_fail = float(np.mean([r["pnr_triggered"] for r in failed_rows]))
        trigger_surv = (
            float(np.mean([r["pnr_triggered"] for r in surv_rows])) if surv_rows else float("nan")
        )
        analysis["pnr_trigger_rate_clean_correct"] = trigger_all
        analysis["pnr_trigger_rate_failed"] = trigger_fail
        analysis["pnr_trigger_rate_survived"] = trigger_surv
        analysis["auc"] = auc_table
        analysis["delta_auc"] = delta_table
        analysis["n_survived"] = len(surv_rows)

    json_path = os.path.join(out_root, "analysis.json")
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)

    config = {
        "experiment": "exp04_pnr_predictive_validity",
        "mode": "mve",
        "seed": seed,
        "n_eval": int(args.n_eval),
        "min_failures": int(args.min_failures),
        "checkpoint": os.path.abspath(args.checkpoint),
        "exp03_json": os.path.abspath(args.exp03_json),
        "shift": asdict(SHIFT_SPEC),
        "frozen": ["checkpoint", "joint_theta", "shift", "layers", "d_norm", "target_class=y"],
        "analysis_population": "clean-correct filter applied after collection",
        "out_root": out_root,
    }
    with open(os.path.join(out_root, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    latest = os.path.join(_REPO_ROOT, "results", "exp04_pnr_predictive_mve.json")
    with open(latest, "w") as f:
        json.dump(analysis, f, indent=2)
    log.info("wrote %s and %s", json_path, latest)
    return analysis


if __name__ == "__main__":
    run(parse_args())
