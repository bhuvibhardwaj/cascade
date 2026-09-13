"""
Statistical layer for population-calibrated PNR thresholds.

Implements Sections 3–5 of "Mathematical Backing for the Population-Calibrated
Point-of-No-Return (PNR) Thresholds". This module is numpy/scipy only; Torch
is imported lazily inside the Cascade-aware calibrators.

What this module establishes
    - DKW ε / ε_sim on empirical quantile estimates θ_k (Section 3 / 3.1).
    - A population-level one-sided Mann–Whitney test of shifted D(k) vs null
      D(k) at a fixed layer (Section 4).
    - BH / BY FDR correction across L layer tests (Section 5).
    - Trajectory-level false-alarm measurement and two fixes for the union
      rule PNR(x) = min{k : D(k) > θ_k} (Section 3.2 Options B and C).

What this module does not establish
    - Individual-level significance of a single input's crossing D(k) > θ_k
      (Section 4.1). MWU is a population test.
    - Causal compounding / mediation / activation patching (Section 7).
    - That naive per-layer 95th-percentile thresholds give a 5% false-PNR
      rate — they do not; use Bonferroni or joint calibration.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Section 3 — DKW
# ---------------------------------------------------------------------------


def dkw_epsilon(n: int, delta: float = 0.05) -> float:
    """Section 3: per-layer DKW deviation ε(n, δ) = sqrt(ln(2/δ) / (2n)).

    If θ_k is the empirical q-quantile of n i.i.d. null D(k) draws, then with
    probability at least 1 − δ,

        F_k(θ_k) ∈ [q − ε, q + ε]   (clipped to [0, 1]).

    This is a single-layer guarantee for one pre-specified k. It does not
    grant simultaneous coverage across all L thresholds (see
    ``dkw_epsilon_simultaneous``), and it does not bound the trajectory-level
    false-alarm rate of the union rule PNR(x) (Section 3.2).
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1).")
    return float(math.sqrt(math.log(2.0 / delta) / (2.0 * n)))


def dkw_epsilon_simultaneous(n: int, n_layers: int, delta: float = 0.05) -> float:
    """Section 3.1: simultaneous DKW bound ε_sim(n, L, δ) = sqrt(ln(2L/δ) / (2n)).

    Union-bound correction: allocate δ/L of the confidence budget to each
    layer so that all L intervals hold at once with probability ≥ 1 − δ.
    Report this bound whenever the L thresholds are presented as a calibrated
    *system*. Shared sampling of the n pairs does not by itself give a joint
    argument.

    This still does not control P(clean trajectory trips PNR anywhere).
    """
    if n_layers <= 0:
        raise ValueError("n_layers must be a positive integer.")
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1).")
    return float(math.sqrt(math.log(2.0 * n_layers / delta) / (2.0 * n)))


# ---------------------------------------------------------------------------
# Section 4 — Mann–Whitney U (shifted D(k) vs null D(k) at a fixed layer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MWUResult:
    """One-sided Mann–Whitney comparison of shifted vs null D(k) at one layer.

    This is a *population* test (Section 4). It is not a p-value for any
    single input's threshold crossing (Section 4.1).
    """

    u_statistic: float
    z_score: float
    p_value: float
    n_null: int
    n_shifted: int
    method: str


def _tie_counts(ranks: np.ndarray) -> np.ndarray:
    _, counts = np.unique(ranks, return_counts=True)
    return counts.astype(float)


def _mwu_normal_approx(
    null_scores: np.ndarray,
    shifted_scores: np.ndarray,
    continuity: bool = True,
) -> Tuple[float, float, float]:
    """Normal approximation with tie correction (Section 4).

    U_shift = R_shift − m(m+1)/2,  mean nm/2,
    σ² = (nm/12) · ((N+1) − Σ(t³−t)/(N(N−1))).

    Continuity correction (subtract 0.5 in the direction that increases the
    p-value) matches ``scipy.stats.mannwhitneyu(..., method='asymptotic')``.
    The math-backing document writes the uncorrected Z; the continuity
    correction is a finite-sample refinement, not a different test.
    """
    n = float(null_scores.size)
    m = float(shifted_scores.size)
    pooled = np.concatenate([shifted_scores, null_scores])
    ranks = stats.rankdata(pooled, method="average")
    r_shift = float(ranks[: shifted_scores.size].sum())
    u = r_shift - m * (m + 1.0) / 2.0
    n_int = n + m
    t = _tie_counts(ranks)
    tie_term = float(np.sum(t**3 - t))
    var = (n * m / 12.0) * ((n_int + 1.0) - tie_term / (n_int * (n_int - 1.0)))
    if var <= 0.0:
        z = 0.0 if abs(u - n * m / 2.0) < 1e-12 else math.copysign(math.inf, u - n * m / 2.0)
        p = 0.5 if z == 0.0 else (0.0 if z > 0.0 else 1.0)
        return u, float(z), float(p)
    sigma = math.sqrt(var)
    mu = n * m / 2.0
    numerator = u - mu
    if continuity:
        # Same sign convention as scipy.stats._mannwhitneyu._get_mwu_z
        # for alternative='greater': shrink |U − μ| toward 0 by 0.5 when
        # U > μ, so the p-value includes mass at the observed U.
        if numerator > 0:
            numerator -= 0.5
        elif numerator < 0:
            numerator += 0.5
    z = numerator / sigma
    p = float(stats.norm.sf(z))
    return float(u), float(z), p


def mann_whitney_layer_test(
    null_scores: Sequence[float],
    shifted_scores: Sequence[float],
) -> MWUResult:
    """Section 4: one-sided MWU, alternative = shifted D(k) stochastically > null D(k).

    Large samples (both groups > 8): continuity-corrected normal approximation
    with tie correction, as written in Section 4 plus the standard ½-correction.

    Small samples (either group ≤ 8): delegates to
    ``scipy.stats.mannwhitneyu(..., alternative='greater', method='auto')``.
    Scipy's ``auto`` is exactly the documented exact/permutation-style fallback:
    it uses the exact U null when both samples are small and there are no ties,
    and the asymptotic (tie-corrected) path otherwise. We do not hand-roll
    that branch.

    Does not establish that any one input's D(k) > θ_k crossing is
    individually significant (Section 4.1).
    """
    null = np.asarray(list(null_scores), dtype=float).ravel()
    shifted = np.asarray(list(shifted_scores), dtype=float).ravel()
    if null.size == 0 or shifted.size == 0:
        raise ValueError("null_scores and shifted_scores must be non-empty.")
    if np.any(~np.isfinite(null)) or np.any(~np.isfinite(shifted)):
        raise ValueError("scores must be finite.")

    n, m = int(null.size), int(shifted.size)
    u, z, p_approx = _mwu_normal_approx(null, shifted, continuity=True)

    # Scipy auto: exact iff both n,m ≤ 8 and no ties; else asymptotic.
    if n <= 8 or m <= 8:
        sc = stats.mannwhitneyu(
            shifted, null, alternative="greater", method="auto"
        )
        return MWUResult(
            u_statistic=float(sc.statistic),
            z_score=float(z),
            p_value=float(sc.pvalue),
            n_null=n,
            n_shifted=m,
            method="scipy_auto",
        )

    return MWUResult(
        u_statistic=u,
        z_score=z,
        p_value=p_approx,
        n_null=n,
        n_shifted=m,
        method="normal_approximation",
    )


# ---------------------------------------------------------------------------
# Section 5 — BH / BY
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BHResult:
    """FDR correction across L per-layer p-values (Section 5)."""

    rejected: List[bool]
    p_adjusted: List[float]
    cutoff_rank: int
    q: float
    method: str


def _fdr_procedure(
    p_values: Sequence[float],
    q: float,
    harmonic: bool,
    method: str,
) -> BHResult:
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        raise ValueError("p_values must be non-empty.")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must lie in [0, 1].")
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0, 1).")

    L = p.size
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    denom = float(np.sum(1.0 / np.arange(1, L + 1))) if harmonic else 1.0
    # Largest i (1-based) with p_(i) ≤ (i/L) * q / H_L
    cutoff = 0
    for i in range(L, 0, -1):
        if p_sorted[i - 1] <= (i / L) * q / denom:
            cutoff = i
            break

    # Adjusted p-values: step-up, truncated at 1.
    raw_adj = np.minimum(1.0, p_sorted * L * denom / np.arange(1, L + 1))
    adj_sorted = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adj = np.empty(L, dtype=float)
    adj[order] = adj_sorted

    rejected = [False] * L
    if cutoff > 0:
        for idx in order[:cutoff]:
            rejected[int(idx)] = True

    return BHResult(
        rejected=rejected,
        p_adjusted=[float(v) for v in adj],
        cutoff_rank=int(cutoff),
        q=float(q),
        method=method,
    )


def benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> BHResult:
    """Section 5: Benjamini–Hochberg FDR. Reject for ranks ≤ largest i with
    p_(i) ≤ (i/L)·q.

    Formal FDR control holds under independence or positive dependence
    (PRDS). Adjacent-layer D(k) is plausibly positively correlated, but that
    is not checked here. If dependence could be negative, use
    ``benjamini_yekutieli`` instead.
    """
    return _fdr_procedure(p_values, q, harmonic=False, method="bh")


def benjamini_yekutieli(p_values: Sequence[float], q: float = 0.05) -> BHResult:
    """Section 5: Benjamini–Yekutieli FDR under arbitrary dependence.

    Same interface as BH, with q replaced by q / H_L, H_L = Σ_{i=1}^L 1/i.
    Conservative fallback when independence / positive dependence cannot be
    assumed. Does not restore individual-level crossing guarantees (4.1).
    """
    return _fdr_procedure(p_values, q, harmonic=True, method="by")


# ---------------------------------------------------------------------------
# Section 3.2 — trajectory false-alarm rate and threshold fixes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FARResult:
    """Empirical union-rule false-alarm count on a null trajectory set.

    ``rate`` is n_triggered / n. Use ``wilson_ci`` / ``format_ci`` rather than
    quoting the bare percentage — at small n the interval is the result.
    """

    n_triggered: int
    n: int
    rate: float

    def wilson_ci(self, alpha: float = 0.05) -> Tuple[float, float]:
        return wilson_proportion_interval(self.n_triggered, self.n, alpha=alpha)

    def format_ci(self, alpha: float = 0.05) -> str:
        lo, hi = self.wilson_ci(alpha=alpha)
        pct = 100.0 * (1.0 - alpha)
        return (
            f"{self.n_triggered}/{self.n} = {100.0 * self.rate:.1f}% "
            f"[{pct:.0f}% CI: {100.0 * lo:.1f}%–{100.0 * hi:.1f}%]"
        )


def wilson_proportion_interval(
    n_success: int, n: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Same interval as ``statsmodels.stats.proportion.proportion_confint(...,
    method='wilson')``. Implemented here so cascade does not depend on
    statsmodels. Prefer Wilson over the normal approximation when n is
    moderate or the rate is near 0 or 1 (the holdout FAR setting).
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if not (0 <= n_success <= n):
        raise ValueError("n_success must be in [0, n].")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p = n_success / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    spread = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    return float(lo), float(hi)


def far_with_ci(
    n_triggered: int, n_holdout: int, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Return (rate, wilson_lo, wilson_hi) for a holdout false-alarm count."""
    if n_holdout <= 0:
        raise ValueError("n_holdout must be a positive integer.")
    rate = n_triggered / n_holdout
    lo, hi = wilson_proportion_interval(n_triggered, n_holdout, alpha=alpha)
    return float(rate), float(lo), float(hi)


def trajectory_false_alarm_rate(
    null_trajectories: np.ndarray,
    thresholds: Sequence[float],
) -> FARResult:
    """Section 3.2: empirical P(a null trajectory trips the union PNR rule).

    ``null_trajectories`` has shape (n_pairs, n_layers). A trajectory trips
    iff any layer satisfies D(k) > θ_k — the same strict inequality used by
    ``diagnose()``.

    Returns counts as well as the ratio so a Wilson interval can be attached.
    Independently setting each θ_k at the 95th percentile yields
    1 − 0.95^L ≈ 0.337 under layer-independence (L = 8), not 0.05. This
    function *measures* that gap; it does not by itself correct it.
    """
    arr = np.asarray(null_trajectories, dtype=float)
    thr = np.asarray(list(thresholds), dtype=float)
    if arr.ndim != 2:
        raise ValueError("null_trajectories must have shape (n_pairs, n_layers).")
    if arr.shape[1] != thr.size:
        raise ValueError(
            f"trajectories have {arr.shape[1]} layers but {thr.size} thresholds."
        )
    if arr.shape[0] == 0:
        raise ValueError("null_trajectories is empty.")
    tripped = np.any(arr > thr.reshape(1, -1), axis=1)
    n_triggered = int(np.sum(tripped))
    n = int(tripped.size)
    return FARResult(n_triggered=n_triggered, n=n, rate=float(n_triggered / n))


def thresholds_at_quantile(trajectories: np.ndarray, quantile: float) -> np.ndarray:
    """Per-layer empirical quantile of an (n_pairs, n_layers) array.

    ``quantile=1`` is allowed and returns the per-layer maximum (needed so
    Option C can drive in-sample union FAR to 0 when n is small).
    """
    if not (0.0 < quantile <= 1.0):
        raise ValueError("quantile must be in (0, 1].")
    arr = np.asarray(trajectories, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("trajectories must have shape (n_pairs, n_layers) with n_pairs>0.")
    return np.quantile(arr, quantile, axis=0)


def calibrate_joint_from_trajectories(
    trajectories: np.ndarray,
    target_fdr: float = 0.05,
    n_search: int = 48,
    base_thresholds: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, float, float]:
    """Section 3.2 Option C, array-level.

    Search approach (documented, deliberately simple): take a per-layer
    threshold *template* (default: the naive ``1 - target_fdr`` quantiles of
    ``trajectories``), then binary-search a single multiplier m so that
    ``trajectory_false_alarm_rate(m · θ_template).rate`` is as close as
    possible to ``target_fdr`` without exceeding it.

    For a three-way split, pass ``base_thresholds`` estimated on the
    calibration fold and ``trajectories`` from a disjoint tuning fold.
    Same-set use (template and FAR both from ``trajectories``) is only for
    unit tests and will overfit the multiplier.

    This is a one-parameter calibration of the existing union rule
    PNR(x) = min{k : D(k) > θ_k}, not Option A's two-stage redefinition.
    When layers are correlated, m stays near 1 and the thresholds are
    tighter than Bonferroni's 1 − α/L quantiles.

    Returns (thresholds, multiplier, measured_far_rate_on_the_search_set).
    """
    if not (0.0 < target_fdr < 1.0):
        raise ValueError("target_fdr must be in (0, 1).")
    arr = np.asarray(trajectories, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("trajectories must have shape (n_pairs, n_layers) with n_pairs>0.")
    if base_thresholds is None:
        base = thresholds_at_quantile(arr, 1.0 - target_fdr)
    else:
        base = np.asarray(list(base_thresholds), dtype=float)
        if base.shape != (arr.shape[1],):
            raise ValueError(
                f"base_thresholds has shape {base.shape}, expected ({arr.shape[1]},)."
            )
    maxima = arr.max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(maxima, np.maximum(base, 1e-15))
    m_hi = float(np.nanmax(ratios))
    if not np.isfinite(m_hi) or m_hi < 1.0:
        m_hi = 1.0
    m_hi = max(m_hi, 1.0) * 1.01 + 1e-9

    far_base = trajectory_false_alarm_rate(arr, base).rate
    if far_base <= target_fdr:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = 1.0, m_hi

    best_m = hi
    best_thr = base * best_m
    best_far = trajectory_false_alarm_rate(arr, best_thr).rate
    for _ in range(n_search):
        mid = 0.5 * (lo + hi)
        thr = base * mid
        far = trajectory_false_alarm_rate(arr, thr).rate
        if far > target_fdr:
            lo = mid
        else:
            hi = mid
            best_m = mid
            best_thr = thr
            best_far = far
    return best_thr, float(best_m), float(best_far)


def _dkw_fields(n_pairs: int, n_layers: int, delta: float = 0.05):
    return {
        "epsilon": dkw_epsilon(n_pairs, delta=delta),
        "epsilon_simultaneous": dkw_epsilon_simultaneous(
            n_pairs, n_layers, delta=delta
        ),
    }


def _collect_pairs(cascade, pairs, n_pairs: Optional[int]) -> np.ndarray:
    """Lazy-import wrapper so this module has no torch import at load time."""
    from .pnr import collect_dk_trajectories

    return collect_dk_trajectories(cascade, pairs, n_pairs=n_pairs)


def calibrate_thresholds_bonferroni(
    cascade,
    pairs: Iterable[Tuple],
    target_fdr: float = 0.05,
    n_pairs: Optional[int] = None,
):
    """Section 3.2 Option B: raise each θ_k to the 1 − α/L percentile.

    Same return type as ``calibrate_pnr_thresholds``. By the union bound,
    P(≥1 false crossing) ≤ target_fdr regardless of dependence; under
    positive correlation this is conservative (lower sensitivity). Does not
    replace DKW ε_sim — that is a different guarantee (threshold estimation
    vs detector FAR).
    """
    from .pnr import PNRThresholds

    if not (0.0 < target_fdr < 1.0):
        raise ValueError("target_fdr must be in (0, 1).")
    traj = _collect_pairs(cascade, pairs, n_pairs)
    n, L = traj.shape
    quantile = 1.0 - target_fdr / L
    values = [float(v) for v in thresholds_at_quantile(traj, quantile)]
    far = trajectory_false_alarm_rate(traj, values)
    return PNRThresholds(
        layer_names=list(cascade.layer_names),
        values=values,
        quantile=float(quantile),
        n_pairs=int(n),
        calibration_method="bonferroni",
        measured_false_alarm_rate=far.rate,
        **_dkw_fields(n, L),
    )


def calibrate_thresholds_joint(
    cascade,
    pairs: Iterable[Tuple],
    target_fdr: float = 0.05,
    n_pairs: Optional[int] = None,
    tune_pairs: Optional[Iterable[Tuple]] = None,
    n_tune_pairs: Optional[int] = None,
):
    """Section 3.2 Option C: empirical joint calibration of the union rule.

    ``pairs`` is the calibration fold: freeze the naive ``1 - target_fdr``
    quantile template. ``tune_pairs`` must be a *disjoint* null fold: search
    the shared multiplier m so ``trajectory_false_alarm_rate`` on tune is
    ≈ ``target_fdr`` (see ``calibrate_joint_from_trajectories``).

    This is the public Option C entry point. Do not skip ``tune_pairs`` and
    treat the result as a quoted FAR — same-fold template + search overfits
    m by construction (in-sample FAR matching). If ``tune_pairs`` is omitted
    (or is the same object as ``pairs``), a runtime warning is raised and
    the call falls back to that same-fold path for tests/smoke checks only.

    ``n_pairs`` on the returned object is the calibration-fold size (DKW ε
    applies to the quantile template). ``measured_false_alarm_rate`` is the
    FAR on the search fold. Quote holdout FAR with a Wilson interval
    (``FARResult.format_ci``), not the tune/in-sample rate.
    """
    from .pnr import PNRThresholds

    if not (0.0 < target_fdr < 1.0):
        raise ValueError("target_fdr must be in (0, 1).")
    calib_traj = _collect_pairs(cascade, pairs, n_pairs)
    n_calib, L = calib_traj.shape
    same_fold = tune_pairs is None or tune_pairs is pairs
    if same_fold:
        warnings.warn(
            "calibrate_thresholds_joint: no disjoint tune_pairs; the Option C "
            "multiplier is fit on the same fold as the percentile template and "
            "will overfit. Pass tune_pairs from a held-out null fold. Same-fold "
            "use is for tests/smoke only — do not cite the in-sample FAR.",
            UserWarning,
            stacklevel=2,
        )
        tune_traj = calib_traj
    else:
        tune_traj = _collect_pairs(cascade, tune_pairs, n_tune_pairs)
        if tune_traj.shape[1] != L:
            raise ValueError(
                f"tune_pairs have {tune_traj.shape[1]} layers, calibration fold has {L}."
            )
    base = thresholds_at_quantile(calib_traj, 1.0 - target_fdr)
    values_arr, _multiplier, far = calibrate_joint_from_trajectories(
        tune_traj, target_fdr=target_fdr, base_thresholds=base
    )
    # `quantile` records the naive template percentile that was then scaled
    # by `multiplier`; the multiplier itself is recoverable as
    # values / naive-quantile if needed. Storing the template percentile
    # keeps the field comparable to the naive/Bonferroni paths.
    return PNRThresholds(
        layer_names=list(cascade.layer_names),
        values=[float(v) for v in values_arr],
        quantile=float(1.0 - target_fdr),
        n_pairs=int(n_calib),
        calibration_method="joint",
        measured_false_alarm_rate=far,
        **_dkw_fields(n_calib, L),
    )
