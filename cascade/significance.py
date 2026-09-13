"""
Per-layer shifted-vs-null significance table (Sections 4–5).

This is the population-level claim: at layer k, the distribution of shifted
D(k) is stochastically larger than the null D(k) distribution. It is not the
paired layer-to-layer test in ``report.py`` (ttest_rel / Wilcoxon on D(k−1)
vs D(k) within a trajectory), and it is not an individual-level p-value for
a single input's PNR crossing (Section 4.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .bounds import (
    BHResult,
    MWUResult,
    benjamini_hochberg,
    benjamini_yekutieli,
    mann_whitney_layer_test,
)
from .pnr import collect_dk_trajectories


@dataclass(frozen=True)
class LayerSignificanceRow:
    layer_name: str
    layer_index: int
    u_statistic: float
    z_score: float
    p_value: float
    p_adjusted: float
    rejected: bool


@dataclass(frozen=True)
class LayerSignificanceTable:
    rows: List[LayerSignificanceRow]
    q: float
    fdr_method: str
    n_null: int
    n_shifted: int
    cutoff_rank: int

    def as_records(self) -> List[dict]:
        return [
            {
                "layer_name": r.layer_name,
                "layer_index": r.layer_index,
                "u_statistic": r.u_statistic,
                "z_score": r.z_score,
                "p_value": r.p_value,
                "p_adjusted": r.p_adjusted,
                "rejected": r.rejected,
            }
            for r in self.rows
        ]


def layer_significance_from_trajectories(
    layer_names: Sequence[str],
    null_traj: np.ndarray,
    shift_traj: np.ndarray,
    q: float = 0.05,
    fdr_method: str = "bh",
) -> LayerSignificanceTable:
    """MWU + FDR from already-computed (n, L) D(k) arrays.

    Use this when trajectories were collected on dedicated MWU folds so we
    do not re-run GradCAM, and so significance does not touch θ_k / FAR data.
    """
    null_traj = np.asarray(null_traj, dtype=float)
    shift_traj = np.asarray(shift_traj, dtype=float)
    if null_traj.ndim != 2 or shift_traj.ndim != 2:
        raise ValueError("trajectories must have shape (n_pairs, n_layers).")
    if null_traj.shape[1] != shift_traj.shape[1]:
        raise ValueError("null and shifted trajectories have different layer counts.")
    names = list(layer_names)
    if len(names) != null_traj.shape[1]:
        raise ValueError("layer_names length does not match trajectory width.")

    mwu_rows: List[MWUResult] = []
    p_values: List[float] = []
    for k in range(null_traj.shape[1]):
        res = mann_whitney_layer_test(null_traj[:, k], shift_traj[:, k])
        mwu_rows.append(res)
        p_values.append(res.p_value)

    method = fdr_method.lower()
    fdr: BHResult
    if method in ("bh", "benjamini-hochberg", "benjamini_hochberg"):
        fdr = benjamini_hochberg(p_values, q=q)
        method_name = "bh"
    elif method in ("by", "benjamini-yekutieli", "benjamini_yekutieli"):
        fdr = benjamini_yekutieli(p_values, q=q)
        method_name = "by"
    else:
        raise ValueError("fdr_method must be 'bh' or 'by'.")

    rows = [
        LayerSignificanceRow(
            layer_name=names[k],
            layer_index=k,
            u_statistic=mwu_rows[k].u_statistic,
            z_score=mwu_rows[k].z_score,
            p_value=mwu_rows[k].p_value,
            p_adjusted=fdr.p_adjusted[k],
            rejected=fdr.rejected[k],
        )
        for k in range(len(names))
    ]
    return LayerSignificanceTable(
        rows=rows,
        q=float(q),
        fdr_method=method_name,
        n_null=int(null_traj.shape[0]),
        n_shifted=int(shift_traj.shape[0]),
        cutoff_rank=fdr.cutoff_rank,
    )


def layer_significance_table(
    cascade,
    null_pairs: Iterable[Tuple],
    shifted_pairs: Iterable[Tuple],
    q: float = 0.05,
    n_null: Optional[int] = None,
    n_shifted: Optional[int] = None,
    fdr_method: str = "bh",
) -> LayerSignificanceTable:
    """MWU at every layer, then BH (default) or BY across the L p-values.

    ``null_pairs`` / ``shifted_pairs`` are independent samples of
    (image_a, image_b, label) — not paired within-trajectory comparisons.
    """
    null_traj = collect_dk_trajectories(cascade, null_pairs, n_pairs=n_null)
    shift_traj = collect_dk_trajectories(cascade, shifted_pairs, n_pairs=n_shifted)
    return layer_significance_from_trajectories(
        cascade.layer_names, null_traj, shift_traj, q=q, fdr_method=fdr_method
    )
