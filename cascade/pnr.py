"""
Population-calibrated Point of No Return (PNR) thresholds.

PNR becomes falsifiable only once "unrecoverable drift" is defined relative
to a null population: how much attribution drift D(k) we expect when there
is no real distribution shift (clean-vs-clean, or clean-vs-benign).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

from .core import Cascade


@dataclass(frozen=True)
class PNRThresholds:
    layer_names: List[str]
    values: List[float]
    quantile: float
    n_pairs: int


def calibrate_pnr_thresholds(
    cascade: Cascade,
    pairs: Iterable[Tuple[torch.Tensor, torch.Tensor, int]],
    quantile: float = 0.95,
    n_pairs: Optional[int] = None,
) -> PNRThresholds:
    """
    Estimate per-layer "unrecoverable" thresholds from a null population.

    pairs: yields (image_a, image_b, label) where (a, b) should represent a
        clean-vs-clean (or otherwise benign) comparison.
    quantile: per-layer cutoff (e.g. 0.95 for a 95th percentile threshold).
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0, 1).")

    all_scores: List[List[float]] = [[] for _ in range(cascade.n_layers)]
    count = 0
    for img_a, img_b, label in pairs:
        _, dk_norm = cascade.dk(img_a, img_b, target_class=label)
        for k, v in enumerate(dk_norm):
            all_scores[k].append(float(v))
        count += 1
        if n_pairs is not None and count >= n_pairs:
            break

    if count == 0:
        raise ValueError("No pairs provided to calibrate_pnr_thresholds().")

    values = [
        float(np.quantile(np.array(layer_scores, dtype=float), quantile))
        for layer_scores in all_scores
    ]

    return PNRThresholds(
        layer_names=list(cascade.layer_names),
        values=values,
        quantile=float(quantile),
        n_pairs=count,
    )

