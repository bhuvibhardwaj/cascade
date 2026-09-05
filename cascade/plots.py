"""
Standard Cascade plots.

These are deliberately simple matplotlib plots with no external dependencies
beyond what the project already requires. Output is saved to a user-supplied
directory.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from typing import List, Optional

import numpy as np

from .report import FragilityProfile


def save_layer_profiles(profile: FragilityProfile, out_dir: str) -> List[str]:
    """Bar/line plots comparing D_norm across layers for each group.

    Produces:
      - layer_dk_norm_mean.png: mean ± 1SD D_norm by layer (both groups)
      - layer_dk_norm_box.png: per-layer boxplot D_norm (both groups)
      - layer_drift.csv      : raw per-sample D(k) records (for downstream analysis)
    """
    os.makedirs(out_dir, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    produced: List[str] = []
    layer_names = profile.layer_names
    xs = np.arange(len(layer_names))

    # (1) mean ± SD line/bar plot
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * len(layer_names)), 4.5))
    colors = {"shifted_incorrect": "#c0392b", "shifted_correct": "#27ae60"}
    for attr, label, grp in (
        ("shifted_incorrect", "shifted-incorrect (failed)", profile.shifted_incorrect),
        ("shifted_correct", "shifted-correct (survived)", profile.shifted_correct),
    ):
        if grp is None:
            continue
        means = grp.true_norm.mean
        stds = grp.true_norm.std
        ax.plot(
            xs, means, marker="o", label=label, color=colors[attr], linewidth=2
        )
        ax.fill_between(
            xs, means - stds, means + stds, color=colors[attr], alpha=0.15
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Layer (depth →)")
    ax.set_ylabel("D_norm(k) — normalized attribution drift")
    ax.set_title("Layer-wise attribution drift: failed vs survived")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "layer_dk_norm_mean.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    produced.append(p)

    # (2) per-layer distribution plot (if we have per-sample records)
    if profile.per_sample:
        groups = {}
        for rec in profile.per_sample:
            groups.setdefault(rec.group, []).append(rec.dk_true_norm)
        if groups:
            n_groups = len(groups)
            n_layers = len(layer_names)
            width = 0.8 / max(1, n_groups)
            fig, ax = plt.subplots(
                figsize=(max(6.0, 0.8 * n_layers), 4.5)
            )
            palette = ["#c0392b", "#27ae60", "#2980b9", "#8e44ad"]
            offsets = (np.arange(n_groups) - n_groups / 2 + 0.5) * width
            for (grp_name, vals), off, col in zip(groups.items(), offsets, palette):
                arr = np.array(vals)
                med = np.median(arr, axis=0)
                q1 = np.quantile(arr, 0.25, axis=0)
                q3 = np.quantile(arr, 0.75, axis=0)
                ax.bar(
                    xs + off,
                    med,
                    width=width,
                    label=grp_name,
                    color=col,
                    alpha=0.85,
                    yerr=[med - q1, q3 - med],
                    capsize=2,
                )
            ax.set_xticks(xs)
            ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("median D_norm(k) [IQR]")
            ax.set_title("Per-layer D_norm by outcome group")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            p = os.path.join(out_dir, "layer_dk_norm_box.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            produced.append(p)

    # (3) per-sample drift CSV
    if profile.per_sample:
        csv_path = os.path.join(out_dir, "layer_drift.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["index", "group", "true_label", "pred_label"]
            for nm in layer_names:
                header.append(f"{nm}_norm")
                header.append(f"{nm}_raw")
            w.writerow(header)
            for rec in profile.per_sample:
                row = [rec.index, rec.group, rec.true_label, rec.pred_label]
                for i in range(len(layer_names)):
                    row.append(rec.dk_true_norm[i])
                    row.append(rec.dk_true_raw[i])
                w.writerow(row)
        produced.append(csv_path)

    return produced


def save_summary_json(profile: FragilityProfile, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload: dict = {"layer_names": profile.layer_names}
    for slot in ("shifted_incorrect", "shifted_correct"):
        grp = getattr(profile, slot)
        if grp is None:
            payload[slot] = None
            continue
        payload[slot] = {
            "n_samples": grp.n_samples,
            "true_norm": {
                "mean": grp.true_norm.mean.tolist(),
                "std": grp.true_norm.std.tolist(),
                "ci_lo": grp.true_norm.ci_lo.tolist(),
                "ci_hi": grp.true_norm.ci_hi.tolist(),
                "median": grp.true_norm.median.tolist(),
            },
            "growth_true_norm": grp.growth_true_norm,
            "ttests_true_rel": [list(x) for x in grp.ttests_true_rel],
            "wilcoxon_true_norm": [list(x) for x in grp.wilcoxon_true_norm],
        }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path
