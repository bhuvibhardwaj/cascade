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
from cascade.metrics import cosine_distance
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
    p.add_argument(
        "--cosine-from-run",
        type=str,
        default=None,
        help=(
            "Existing Exp 04 run directory. Populate maps_cache via dk("
            "return_maps=True) if missing, then write samples_cos.csv / "
            "analysis_cos.json. Does not reuse D_norm joint θ for cosine PNR."
        ),
    )
    p.add_argument(
        "--pred-from-run",
        type=str,
        default=None,
        help=(
            "Same eval indices as an Exp 04 run, but GradCAM target is the "
            "shifted predicted class (layer_drift dk_pred), not the true label. "
            "Writes maps_cache_pred/, samples_pred.csv, analysis_pred.json."
        ),
    )
    return p.parse_args()


def _detach_maps(maps):
    return [t.detach().float().cpu().clone() for t in maps]


def _maps_cache_path(run_dir: str, dataset_index: int) -> str:
    return os.path.join(run_dir, "maps_cache", f"{int(dataset_index)}.pt")


def _load_sample_rows(csv_path: str):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes"}


def run_cosine_from_run(args):
    """Second metric on the same eval indices/outcomes as an Exp 04 L2 run.

    Exp 03/04 only persisted scalar D_raw / D_norm. GradCAM maps were computed
    inside dk() and discarded. This pass keeps A_clean / A_shift on disk, then
    scores D_cos(k) = 1 - cosine_similarity from those tensors.

    Joint θ from Exp 03 is a D_norm threshold. It is not applied to D_cos.
    """
    run_dir = os.path.abspath(args.cosine_from_run)
    samples_path = os.path.join(run_dir, "samples.csv")
    splits_path = os.path.join(run_dir, "splits.json")
    if not os.path.isfile(samples_path):
        raise FileNotFoundError(f"missing {samples_path}")
    if not os.path.isfile(splits_path):
        raise FileNotFoundError(f"missing {splits_path}")

    with open(splits_path) as f:
        splits = json.load(f)
    eval_indices = [int(i) for i in splits["eval_indices"]]
    rows_l2 = _load_sample_rows(samples_path)
    by_idx = {int(r["dataset_index"]): r for r in rows_l2}
    missing = [i for i in eval_indices if i not in by_idx]
    if missing:
        raise RuntimeError(f"samples.csv missing indices {missing[:5]}...")

    cache_dir = os.path.join(run_dir, "maps_cache")
    os.makedirs(cache_dir, exist_ok=True)
    need = [i for i in eval_indices if not os.path.isfile(_maps_cache_path(run_dir, i))]

    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device()

    log_path = os.path.join(run_dir, "cosine.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    log = logging.getLogger("exp04_cos")
    log.info("cosine pass on %s", run_dir)
    log.info("eval n=%d; maps to compute=%d (rest cached)", len(eval_indices), len(need))
    log.info(
        "D_norm joint θ will NOT be applied to D_cos (different scale/metric)."
    )

    if need:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(
                f"Exp 03 checkpoint not found: {args.checkpoint}. Do not retrain."
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
        log.info("layers: %s", cascade.layer_names)

        t0 = time.time()
        for j, idx in enumerate(need):
            rec = by_idx[idx]
            y = int(rec["true_label"])
            x_raw, _ = test_raw[idx]
            if not isinstance(x_raw, torch.Tensor):
                x01 = to_tensor(x_raw)
            else:
                x01 = x_raw
            x, _ = test_clean[idx]
            xs = norm(shift_fn(x01))
            x_batch = x.unsqueeze(0) if x.dim() == 3 else x
            xs_batch = xs.unsqueeze(0) if xs.dim() == 3 else xs
            x_batch = x_batch.to(device)
            xs_batch = xs_batch.to(device)

            d_raw, d_norm, a_clean, a_shift = cascade.dk(
                x_batch, xs_batch, target_class=y, return_maps=True
            )
            payload = {
                "dataset_index": idx,
                "true_label": y,
                "layer_names": list(cascade.layer_names),
                "A_clean": _detach_maps(a_clean),
                "A_shift": _detach_maps(a_shift),
                "d_raw_from_maps": [float(v) for v in d_raw],
                "d_norm_from_maps": [float(v) for v in d_norm],
            }
            torch.save(payload, _maps_cache_path(run_dir, idx))

            done = j + 1
            n = len(need)
            if done == 10 or done % 50 == 0 or done == n:
                elapsed = time.time() - t0
                rate = elapsed / done
                log.info(
                    "cache %d/%d  %.2fs/pair  ETA %.0fs",
                    done, n, rate, rate * (n - done),
                )
    else:
        log.info("maps_cache already complete; skipping GradCAM")

    fieldnames = [
        "dataset_index",
        "true_label",
        "pred_clean",
        "pred_shift",
        "correct_clean",
        "correct_shift",
        "failed",
        "conf_shift",
        *[f"d_cos_{k}" for k in range(MAX_LAYERS)],
        "max_d_cos",
        "argmax_d_cos",
        "d_last_cos",
        "d_early_mean_cos",
        "max_d_norm",
        "d_last_norm",
    ]
    cos_rows = []
    raw_mismatch = 0
    for idx in eval_indices:
        rec = by_idx[idx]
        payload = torch.load(_maps_cache_path(run_dir, idx), map_location="cpu")
        a_clean = payload["A_clean"]
        a_shift = payload["A_shift"]
        if len(a_clean) != MAX_LAYERS:
            raise RuntimeError(
                f"cache {idx}: expected {MAX_LAYERS} maps, got {len(a_clean)}"
            )
        d_cos = [
            cosine_distance(ac, ash) for ac, ash in zip(a_clean, a_shift)
        ]
        cached_raw = payload.get("d_raw_from_maps")
        if cached_raw is not None:
            for k in range(MAX_LAYERS):
                prev = float(rec[f"d_raw_{k}"])
                if abs(prev - float(cached_raw[k])) > 1e-3:
                    raw_mismatch += 1
                    break

        row = {
            "dataset_index": idx,
            "true_label": int(rec["true_label"]),
            "pred_clean": int(rec["pred_clean"]),
            "pred_shift": int(rec["pred_shift"]),
            "correct_clean": _as_bool(rec["correct_clean"]),
            "correct_shift": _as_bool(rec["correct_shift"]),
            "failed": _as_bool(rec["failed"]),
            "conf_shift": float(rec["conf_shift"]),
            "max_d_cos": float(max(d_cos)),
            "argmax_d_cos": int(int(np.argmax(d_cos))),
            "d_last_cos": float(d_cos[-1]),
            "d_early_mean_cos": float(np.mean(d_cos[:2])),
            "max_d_norm": float(rec["max_d"]),
            "d_last_norm": float(rec["d_last"]),
        }
        for k in range(MAX_LAYERS):
            row[f"d_cos_{k}"] = float(d_cos[k])
        cos_rows.append(row)

    if raw_mismatch:
        log.warning(
            "D_raw from recached maps differed from samples.csv on %d/%d "
            "indices (float/device noise). Outcomes still taken from samples.csv.",
            raw_mismatch,
            len(eval_indices),
        )

    csv_path = os.path.join(run_dir, "samples_cos.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(cos_rows)
    log.info("wrote %s", csv_path)

    eval_rows = [r for r in cos_rows if r["correct_clean"]]
    n_cc = len(eval_rows)
    n_fail = sum(1 for r in eval_rows if r["failed"])
    n_surv = n_cc - n_fail
    log.info("clean-correct: %d  failures: %d  survived: %d", n_cc, n_fail, n_surv)

    analysis = {
        "experiment": "exp04_pnr_predictive_validity",
        "mode": "mve_cosine",
        "source_run": run_dir,
        "metric": "d_cos",
        "d_cos_definition": "1 - cosine_similarity(flatten(A_shift), flatten(A_clean))",
        "pnr_applied": False,
        "pnr_note": (
            "Exp 03 joint θ is calibrated on D_norm. It is not a cosine "
            "threshold and was not used."
        ),
        "n_eval_raw": len(cos_rows),
        "n_clean_correct": n_cc,
        "n_failures": n_fail,
        "n_survived": n_surv,
        "maps_cache": cache_dir,
        "d_raw_cache_vs_csv_mismatched_indices": raw_mismatch,
    }

    if n_fail < int(args.min_failures):
        analysis["aborted"] = True
        analysis["abort_reason"] = (
            f"clean-correct failures {n_fail} < {args.min_failures}"
        )
        log.warning("%s", analysis["abort_reason"])
    else:
        analysis["aborted"] = False
        y = np.array([1 if r["failed"] else 0 for r in eval_rows], dtype=int)
        scores = {
            "max_d_cos": np.array([r["max_d_cos"] for r in eval_rows], dtype=float),
            "d_last_cos": np.array([r["d_last_cos"] for r in eval_rows], dtype=float),
            "d_early_mean_cos": np.array(
                [r["d_early_mean_cos"] for r in eval_rows], dtype=float
            ),
            "max_d_norm": np.array([r["max_d_norm"] for r in eval_rows], dtype=float),
            "d_last_norm": np.array([r["d_last_norm"] for r in eval_rows], dtype=float),
            "neg_conf_shift": np.array(
                [-r["conf_shift"] for r in eval_rows], dtype=float
            ),
        }
        boot_rng = np.random.default_rng(seed + 1)
        auc_table = {}
        for name, sc in scores.items():
            point, lo, hi = bootstrap_auc(y, sc, int(args.n_boot), boot_rng)
            auc_table[name] = {"auc": point, "ci95_lo": lo, "ci95_hi": hi}
            log.info("AUC %s: %.4f [%.4f, %.4f]", name, point, lo, hi)
        delta_table = {}
        for other in ("neg_conf_shift", "max_d_norm", "d_last_cos"):
            d, lo, hi = bootstrap_delta_auc(
                y, scores["max_d_cos"], scores[other], int(args.n_boot), boot_rng
            )
            delta_table[f"max_d_cos_minus_{other}"] = {
                "delta_auc": d,
                "ci95_lo": lo,
                "ci95_hi": hi,
            }
            log.info("ΔAUC max_d_cos - %s: %.4f [%.4f, %.4f]", other, d, lo, hi)
        analysis["auc"] = auc_table
        analysis["delta_auc"] = delta_table
        fail_rows = [r for r in eval_rows if r["failed"]]
        surv_rows = [r for r in eval_rows if not r["failed"]]
        analysis["mean_max_d_cos_failed"] = float(
            np.mean([r["max_d_cos"] for r in fail_rows])
        )
        analysis["mean_max_d_cos_survived"] = float(
            np.mean([r["max_d_cos"] for r in surv_rows])
        ) if surv_rows else float("nan")

    json_path = os.path.join(run_dir, "analysis_cos.json")
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)
    latest = os.path.join(_REPO_ROOT, "results", "exp04_pnr_predictive_mve_cos.json")
    with open(latest, "w") as f:
        json.dump(analysis, f, indent=2)
    log.info("wrote %s and %s", json_path, latest)
    return analysis


def run_pred_from_run(args):
    """True-label maps are already cached. This pass uses pred_shift as GradCAM target.

    Survivors have pred_shift == y, so pred-label maps must match true-label maps.
    Failures have pred_shift != y; that is the independent comparison.
    """
    run_dir = os.path.abspath(args.pred_from_run)
    samples_path = os.path.join(run_dir, "samples.csv")
    splits_path = os.path.join(run_dir, "splits.json")
    if not os.path.isfile(samples_path) or not os.path.isfile(splits_path):
        raise FileNotFoundError(run_dir)

    with open(splits_path) as f:
        splits = json.load(f)
    eval_indices = [int(i) for i in splits["eval_indices"]]
    by_idx = {int(r["dataset_index"]): r for r in _load_sample_rows(samples_path)}

    cache_dir = os.path.join(run_dir, "maps_cache_pred")
    os.makedirs(cache_dir, exist_ok=True)

    def cache_path(i):
        return os.path.join(cache_dir, f"{int(i)}.pt")

    need = [i for i in eval_indices if not os.path.isfile(cache_path(i))]
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device()

    log_path = os.path.join(run_dir, "pred.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    log = logging.getLogger("exp04_pred")
    log.info("pred-label GradCAM on %s  need=%d/%d", run_dir, len(need), len(eval_indices))

    if need:
        data_root = os.path.join(_REPO_ROOT, "data")
        test_raw = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=None
        )
        test_clean = NormWrapper(test_raw)
        shift_fn = build_shift(SHIFT_SPEC)
        to_tensor = T.ToTensor()
        norm = T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
        model = build_cifar_resnet18().to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        cascade = Cascade(model, max_layers=MAX_LAYERS, device=str(device))
        t0 = time.time()
        for j, idx in enumerate(need):
            rec = by_idx[idx]
            pred_shift = int(rec["pred_shift"])
            x_raw, _ = test_raw[idx]
            x01 = x_raw if isinstance(x_raw, torch.Tensor) else to_tensor(x_raw)
            x, _ = test_clean[idx]
            xs = norm(shift_fn(x01))
            x_batch = (x.unsqueeze(0) if x.dim() == 3 else x).to(device)
            xs_batch = (xs.unsqueeze(0) if xs.dim() == 3 else xs).to(device)
            d_raw, d_norm, a_clean, a_shift = cascade.dk(
                x_batch, xs_batch, target_class=pred_shift, return_maps=True
            )
            torch.save(
                {
                    "dataset_index": idx,
                    "target_class": pred_shift,
                    "layer_names": list(cascade.layer_names),
                    "A_clean": _detach_maps(a_clean),
                    "A_shift": _detach_maps(a_shift),
                    "d_raw_from_maps": [float(v) for v in d_raw],
                    "d_norm_from_maps": [float(v) for v in d_norm],
                },
                cache_path(idx),
            )
            done = j + 1
            if done == 10 or done % 50 == 0 or done == len(need):
                elapsed = time.time() - t0
                log.info("cache %d/%d  %.2fs/pair", done, len(need), elapsed / done)

    fieldnames = [
        "dataset_index", "true_label", "pred_shift", "correct_clean", "failed",
        "target_equals_true",
        *[f"d_cos_pred_{k}" for k in range(MAX_LAYERS)],
        *[f"d_norm_pred_{k}" for k in range(MAX_LAYERS)],
        "max_d_cos_pred", "d_last_cos_pred", "d_early_mean_cos_pred",
        "max_d_norm_pred", "d_last_norm_pred",
        "max_d_cos_true", "max_d_norm_true",
    ]
    cos_true = {}
    cos_path = os.path.join(run_dir, "samples_cos.csv")
    if os.path.isfile(cos_path):
        for r in _load_sample_rows(cos_path):
            cos_true[int(r["dataset_index"])] = r

    pred_rows = []
    for idx in eval_indices:
        rec = by_idx[idx]
        payload = torch.load(cache_path(idx), map_location="cpu")
        d_cos = [
            cosine_distance(ac, ash)
            for ac, ash in zip(payload["A_clean"], payload["A_shift"])
        ]
        d_norm = [float(v) for v in payload["d_norm_from_maps"]]
        y = int(rec["true_label"])
        pred_shift = int(rec["pred_shift"])
        ct = cos_true.get(idx, {})
        row = {
            "dataset_index": idx,
            "true_label": y,
            "pred_shift": pred_shift,
            "correct_clean": _as_bool(rec["correct_clean"]),
            "failed": _as_bool(rec["failed"]),
            "target_equals_true": pred_shift == y,
            "max_d_cos_pred": float(max(d_cos)),
            "d_last_cos_pred": float(d_cos[-1]),
            "d_early_mean_cos_pred": float(np.mean(d_cos[:2])),
            "max_d_norm_pred": float(max(d_norm)),
            "d_last_norm_pred": float(d_norm[-1]),
            "max_d_cos_true": float(ct["max_d_cos"]) if ct else float("nan"),
            "max_d_norm_true": float(rec["max_d"]),
        }
        for k in range(MAX_LAYERS):
            row[f"d_cos_pred_{k}"] = float(d_cos[k])
            row[f"d_norm_pred_{k}"] = d_norm[k]
        pred_rows.append(row)

    csv_path = os.path.join(run_dir, "samples_pred.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(pred_rows)
    log.info("wrote %s", csv_path)

    eval_rows = [r for r in pred_rows if r["correct_clean"]]
    fail_rows = [r for r in eval_rows if r["failed"]]
    surv_rows = [r for r in eval_rows if not r["failed"]]
    surv_labels = sorted(set(r["true_label"] for r in surv_rows))
    fail_labels = sorted(set(r["true_label"] for r in fail_rows))
    log.info("survivors n=%d labels=%s", len(surv_rows), surv_labels)
    log.info("failures n=%d labels=%s", len(fail_rows), fail_labels)

    y = np.array([1 if r["failed"] else 0 for r in eval_rows], dtype=int)
    scores = {
        "max_d_cos_pred": np.array([r["max_d_cos_pred"] for r in eval_rows]),
        "d_last_cos_pred": np.array([r["d_last_cos_pred"] for r in eval_rows]),
        "max_d_norm_pred": np.array([r["max_d_norm_pred"] for r in eval_rows]),
        "d_last_norm_pred": np.array([r["d_last_norm_pred"] for r in eval_rows]),
        "max_d_cos_true": np.array([r["max_d_cos_true"] for r in eval_rows]),
        "max_d_norm_true": np.array([r["max_d_norm_true"] for r in eval_rows]),
    }
    boot_rng = np.random.default_rng(seed + 1)
    auc_table = {}
    for name, sc in scores.items():
        if not np.isfinite(sc).all():
            continue
        point, lo, hi = bootstrap_auc(y, sc, int(args.n_boot), boot_rng)
        auc_table[name] = {"auc": point, "ci95_lo": lo, "ci95_hi": hi}
        log.info("AUC %s: %.4f [%.4f, %.4f]", name, point, lo, hi)

    # Survivors vs fail: pred maps must match true maps on survivors.
    surv_match = []
    for r in surv_rows:
        if np.isfinite(r["max_d_cos_true"]):
            surv_match.append(abs(r["max_d_cos_pred"] - r["max_d_cos_true"]))
    analysis = {
        "experiment": "exp04_pnr_predictive_validity",
        "mode": "mve_pred_label",
        "source_run": run_dir,
        "gradcam_target": "pred_shift",
        "n_clean_correct": len(eval_rows),
        "n_failures": len(fail_rows),
        "n_survived": len(surv_rows),
        "survivor_true_labels": surv_labels,
        "failure_true_labels": fail_labels,
        "class_confound_note": (
            "If survivors are a single CIFAR class and that class is absent "
            "from failures, fail-vs-survive D(k) is a class comparison."
        ),
        "survivor_max_d_cos_pred_minus_true_max_abs": (
            float(max(surv_match)) if surv_match else float("nan")
        ),
        "mean_max_d_cos_pred_failed": float(np.mean([r["max_d_cos_pred"] for r in fail_rows])),
        "mean_max_d_cos_pred_survived": float(np.mean([r["max_d_cos_pred"] for r in surv_rows])),
        "mean_max_d_norm_pred_failed": float(np.mean([r["max_d_norm_pred"] for r in fail_rows])),
        "mean_max_d_norm_pred_survived": float(np.mean([r["max_d_norm_pred"] for r in surv_rows])),
        "auc": auc_table,
    }
    json_path = os.path.join(run_dir, "analysis_pred.json")
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)
    log.info("wrote %s", json_path)
    return analysis


def run(args):
    if args.pred_from_run:
        return run_pred_from_run(args)
    if args.cosine_from_run:
        return run_cosine_from_run(args)

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
