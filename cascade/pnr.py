"""
Population-calibrated Point of No Return (PNR) thresholds.

PNR becomes falsifiable only once "unrecoverable drift" is defined relative
to a null population: how much attribution drift D(k) we expect when there
is no real distribution shift (clean-vs-clean, or clean-vs-benign).

``calibrate_pnr_thresholds()`` is the naive per-layer quantile path (Section 3
of the math-backing doc). It does **not** by itself give a 5% trajectory-level
false-alarm rate for the union rule PNR(x) = min{k : D(k) > θ_k} — see
``cascade.bounds.calibrate_thresholds_bonferroni`` (Option B) and
``cascade.bounds.calibrate_thresholds_joint`` (Option C; pass disjoint
``tune_pairs`` — same-fold use overfits the multiplier).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

from .bounds import (
    dkw_epsilon,
    dkw_epsilon_simultaneous,
    trajectory_false_alarm_rate,
)
from .core import Cascade


@dataclass(frozen=True)
class PNRThresholds:
    layer_names: List[str]
    values: List[float]
    quantile: float
    n_pairs: int
    epsilon: Optional[float] = None
    epsilon_simultaneous: Optional[float] = None
    calibration_method: str = "naive"
    measured_false_alarm_rate: Optional[float] = None


def collect_dk_trajectories(
    cascade: Cascade,
    pairs: Iterable[Tuple[torch.Tensor, torch.Tensor, int]],
    n_pairs: Optional[int] = None,
) -> np.ndarray:
    """Return an (n_pairs, n_layers) array of D_norm(k) trajectories."""
    rows: List[List[float]] = []
    count = 0
    for img_a, img_b, label in pairs:
        _, dk_norm = cascade.dk(img_a, img_b, target_class=label)
        rows.append([float(v) for v in dk_norm])
        count += 1
        if n_pairs is not None and count >= n_pairs:
            break
    if count == 0:
        raise ValueError("No pairs provided to collect_dk_trajectories().")
    arr = np.asarray(rows, dtype=float)
    if arr.shape[1] != cascade.n_layers:
        raise ValueError(
            f"Expected {cascade.n_layers} layers, got {arr.shape[1]} D(k) values."
        )
    return arr


def calibrate_pnr_thresholds(
    cascade: Cascade,
    pairs: Iterable[Tuple[torch.Tensor, torch.Tensor, int]],
    quantile: float = 0.95,
    n_pairs: Optional[int] = None,
) -> PNRThresholds:
    """
    Estimate per-layer "unrecoverable" thresholds from a null population.

    This is the **naive** path: each θ_k is the empirical ``quantile`` of that
    layer's null D(k), with no Bonferroni or joint-trajectory correction.
    Independently using quantile=0.95 at L=8 layers gives ~33.7% probability
    that a clean trajectory trips the union rule somewhere (Section 3.2).

    pairs: yields (image_a, image_b, label) where (a, b) should represent a
        clean-vs-clean (or otherwise benign) comparison.
    quantile: per-layer cutoff (e.g. 0.95 for a 95th percentile threshold).
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0, 1).")

    traj = collect_dk_trajectories(cascade, pairs, n_pairs=n_pairs)
    n, L = traj.shape
    values = [float(v) for v in np.quantile(traj, quantile, axis=0)]
    far = trajectory_false_alarm_rate(traj, values)

    return PNRThresholds(
        layer_names=list(cascade.layer_names),
        values=values,
        quantile=float(quantile),
        n_pairs=int(n),
        epsilon=dkw_epsilon(n),
        epsilon_simultaneous=dkw_epsilon_simultaneous(n, L),
        calibration_method="naive",
        measured_false_alarm_rate=far.rate,
    )
