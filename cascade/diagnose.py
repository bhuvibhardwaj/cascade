"""
Per-input diagnostics.

This is the piece with no notebook precedent: instead of only reporting
aggregate D(k) statistics across many samples, `diagnose()` looks at a
single misclassified input and answers two questions:

1. Point of no return — which layer is the first one where this input's
   attribution drift crosses an "unrecoverable" threshold?
2. Stable vs unstable — does the model's own (possibly wrong) reasoning
   path drift in lockstep with the correct semantic path, or do they
   diverge? A "stable" failure is confidently, consistently wrong; an
   "unstable" one is flickering near the decision boundary. These need
   different fixes (stable failures need better features / more data on
   that shift; unstable ones may just need calibration or a rejection
   threshold).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from .core import Cascade, LayerDrift
from .pnr import PNRThresholds


@dataclass
class DiagnosisResult:
    layer_names: List[str]
    dk_true: List[float]
    dk_pred: List[float]
    point_of_no_return: Optional[int]  # index into layer_names, or None
    point_of_no_return_layer: Optional[str]
    pnr_thresholds: Optional[List[float]]
    correlation: float
    verdict: str  # "stable" or "unstable"
    calibration_method: Optional[str] = None
    epsilon: Optional[float] = None
    epsilon_simultaneous: Optional[float] = None
    measured_false_alarm_rate: Optional[float] = None

    def summary(self) -> str:
        if self.pnr_thresholds is None:
            pnr = "not computed (no calibrated thresholds provided)"
        elif self.point_of_no_return is None:
            pnr = "not reached within instrumented layers"
        else:
            pnr = f"{self.point_of_no_return_layer} (layer {self.point_of_no_return})"
        lines = [
            f"Point of no return: {pnr}",
            f"True/predicted drift correlation: {self.correlation:.3f}",
            f"Verdict: {self.verdict}",
        ]
        if self.pnr_thresholds is not None:
            method = self.calibration_method or "unspecified"
            lines.append(f"PNR calibration method: {method}")
            if method == "naive":
                lines.append(
                    "  (naive per-layer quantiles do not give a 5% trajectory "
                    "false-alarm rate; see Bonferroni / joint calibration)"
                )
            if self.epsilon is not None:
                lines.append(f"DKW ε (single layer): {self.epsilon:.4f}")
            if self.epsilon_simultaneous is not None:
                lines.append(
                    f"DKW ε_sim (all {len(self.layer_names)} layers): "
                    f"{self.epsilon_simultaneous:.4f}"
                )
            if self.measured_false_alarm_rate is not None:
                lines.append(
                    f"Measured null trajectory false-alarm rate: "
                    f"{self.measured_false_alarm_rate:.4f}"
                )
        return "\n".join(lines)


def _auto_threshold(dk_true: List[float]) -> float:
    """
    Self-referential heuristic threshold: mean + 1 std of this sample's own
    D(k) trajectory.

    This is useful for debugging but should not be used to make claims about
    "unrecoverable" drift, since it is not calibrated against a clean
    population baseline.
    """
    arr = np.array(dk_true)
    return float(arr.mean() + arr.std())


ThresholdSpec = Union[float, Sequence[float], PNRThresholds]


def _resolve_thresholds(
    threshold: Optional[ThresholdSpec], n_layers: int
) -> Optional[List[float]]:
    if threshold is None:
        return None

    if isinstance(threshold, (int, float)):
        return [float(threshold) for _ in range(n_layers)]

    if isinstance(threshold, PNRThresholds):
        if len(threshold.values) != n_layers:
            raise ValueError(
                f"PNRThresholds has {len(threshold.values)} values but cascade has "
                f"{n_layers} layers."
            )
        return [float(v) for v in threshold.values]

    if len(threshold) != n_layers:
        raise ValueError(
            f"threshold has length {len(threshold)} but cascade has {n_layers} layers."
        )
    return [float(v) for v in threshold]


def diagnose(
    cascade: Cascade,
    clean_image: torch.Tensor,
    shifted_image: torch.Tensor,
    true_label: int,
    pred_label: int,
    threshold: Optional[ThresholdSpec] = None,
    stability_corr_cutoff: float = 0.8,
) -> DiagnosisResult:
    """
    Diagnose a single (clean, shifted) input pair.

    threshold: population-calibrated cutoff defining "unrecoverable" drift.
        Pass either:
        - a scalar float used at all layers, or
        - a per-layer sequence of floats, or
        - a PNRThresholds object from calibrate_pnr_thresholds (naive),
          calibrate_thresholds_bonferroni, or calibrate_thresholds_joint.
          Option C is not interchangeable with the first two unless you pass
          disjoint ``tune_pairs``; omitting them overfits the multiplier (the
          function warns). The object's calibration_method, epsilon, and
          epsilon_simultaneous are copied onto the result so naive thresholds
          cannot be presented as Bonferroni/joint-corrected.
        If None, PNR is not computed (but the stable/unstable verdict is).
    stability_corr_cutoff: correlation above which the true-label and
        predicted-label drift trajectories are considered to track each
        other closely ("stable" failure) rather than diverge ("unstable").
    """
    drift: LayerDrift = cascade.layer_drift(
        clean_image, shifted_image, true_label, pred_label
    )

    thresholds = _resolve_thresholds(threshold, n_layers=len(drift.dk_true))
    calib_method = threshold.calibration_method if isinstance(threshold, PNRThresholds) else None
    epsilon = threshold.epsilon if isinstance(threshold, PNRThresholds) else None
    epsilon_sim = (
        threshold.epsilon_simultaneous if isinstance(threshold, PNRThresholds) else None
    )
    measured_far = (
        threshold.measured_false_alarm_rate
        if isinstance(threshold, PNRThresholds)
        else None
    )
    pnr_idx = (
        next((i for i, d in enumerate(drift.dk_true) if d > thresholds[i]), None)
        if thresholds is not None
        else None
    )
    pnr_layer = drift.layer_names[pnr_idx] if pnr_idx is not None else None

    if len(drift.dk_true) > 1 and np.std(drift.dk_true) > 0 and np.std(drift.dk_pred) > 0:
        correlation = float(np.corrcoef(drift.dk_true, drift.dk_pred)[0, 1])
    else:
        correlation = float("nan")

    verdict = (
        "stable"
        if not np.isnan(correlation) and correlation > stability_corr_cutoff
        else "unstable"
    )

    return DiagnosisResult(
        layer_names=drift.layer_names,
        dk_true=drift.dk_true,
        dk_pred=drift.dk_pred,
        point_of_no_return=pnr_idx,
        point_of_no_return_layer=pnr_layer,
        pnr_thresholds=thresholds,
        correlation=correlation,
        verdict=verdict,
        calibration_method=calib_method,
        epsilon=epsilon,
        epsilon_simultaneous=epsilon_sim,
        measured_false_alarm_rate=measured_far,
    )
