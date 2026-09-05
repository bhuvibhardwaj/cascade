"""
Aggregate layer-wise attribution-drift profiles.

Key changes vs the original:
  - Measurements are paired (same sample at layer k-1 and k): use ttest_rel
    with a Wilcoxon signed-rank non-parametric fallback.
  - fragility_profile() accepts both shifted-CORRECT and shifted-INCORRECT
    samples so drift trajectories can be compared between the survived group
    and the failed group.
  - Reports both raw D(k) and normalized D(k) (D_norm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy import stats

from .core import Cascade, LayerDrift
from .samples import Sample, SampleDrift


@dataclass
class LayerStats:
    mean: np.ndarray
    std: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    median: np.ndarray


@dataclass
class GroupProfile:
    name: str
    n_samples: int
    true_raw: LayerStats
    true_norm: LayerStats
    pred_raw: LayerStats
    pred_norm: LayerStats
    growth_true_norm: List[float]
    ttests_true_rel: List[Tuple[float, float]]
    wilcoxon_true_norm: List[Tuple[float, float]]


@dataclass
class FragilityProfile:
    layer_names: List[str]
    shifted_incorrect: Optional[GroupProfile] = None
    shifted_correct: Optional[GroupProfile] = None
    per_sample: List[SampleDrift] = field(default_factory=list)

    def summary(self) -> str:
        lines: List[str] = [f"Layers: {self.layer_names}", ""]
        for group in (self.shifted_incorrect, self.shifted_correct):
            if group is None:
                continue
            lines.append(
                f"[{group.name}] n={group.n_samples} — D_norm (true label):"
            )
            for i, name in enumerate(self.layer_names):
                lines.append(
                    f"  {name}: mean={group.true_norm.mean[i]:.4f} "
                    f"med={group.true_norm.median[i]:.4f} "
                    f"±{group.true_norm.std[i]:.4f} "
                    f"CI=[{group.true_norm.ci_lo[i]:.4f}, "
                    f"{group.true_norm.ci_hi[i]:.4f}]"
                )
            lines.append("  Adjacent-layer paired t (norm-true):")
            for i, (t, p) in enumerate(group.ttests_true_rel):
                sig = "*" if p < 0.05 else " "
                lines.append(
                    f"    {self.layer_names[i + 1]} vs {self.layer_names[i]}: "
                    f"t={t:+.3f} p={p:.4f}{sig}"
                )
            lines.append("  Wilcoxon signed-rank (norm-true):")
            for i, (w, p) in enumerate(group.wilcoxon_true_norm):
                sig = "*" if p < 0.05 else " "
                lines.append(
                    f"    {self.layer_names[i + 1]} vs {self.layer_names[i]}: "
                    f"W={w:+.1f} p={p:.4f}{sig}"
                )
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _summarise(scores: List[List[float]]) -> LayerStats:
    means, stds, ci_lo, ci_hi, meds = [], [], [], [], []
    for layer_scores in scores:
        arr = np.array(layer_scores, dtype=float)
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        med = float(np.median(arr))
        n = len(arr)
        if n > 1:
            se = s / np.sqrt(n)
            t_crit = stats.t.ppf(0.975, df=n - 1)
            lo, hi = m - t_crit * se, m + t_crit * se
        else:
            lo = hi = m
        means.append(m)
        stds.append(s)
        ci_lo.append(lo)
        ci_hi.append(hi)
        meds.append(med)
    return LayerStats(
        mean=np.array(means),
        std=np.array(stds),
        ci_lo=np.array(ci_lo),
        ci_hi=np.array(ci_hi),
        median=np.array(meds),
    )


def _growth(means: np.ndarray) -> List[float]:
    return [
        means[i] / means[i - 1] if means[i - 1] != 0 else float("nan")
        for i in range(1, len(means))
    ]


def _paired_ttests_rel(scores: List[List[float]]) -> List[Tuple[float, float]]:
    results: List[Tuple[float, float]] = []
    for k in range(1, len(scores)):
        a = np.array(scores[k], dtype=float)
        b = np.array(scores[k - 1], dtype=float)
        n = min(len(a), len(b))
        if n < 2:
            results.append((float("nan"), float("nan")))
            continue
        t, p = stats.ttest_rel(a[:n], b[:n])
        results.append((float(t), float(p)))
    return results


def _wilcoxon(scores: List[List[float]]) -> List[Tuple[float, float]]:
    results: List[Tuple[float, float]] = []
    for k in range(1, len(scores)):
        a = np.array(scores[k], dtype=float)
        b = np.array(scores[k - 1], dtype=float)
        diffs = a[: min(len(a), len(b))] - b[: min(len(a), len(b))]
        if len(diffs) < 3 or np.allclose(diffs, 0.0):
            results.append((float("nan"), float("nan")))
            continue
        try:
            res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            results.append((float(res.statistic), float(res.pvalue)))
        except ValueError:
            results.append((float("nan"), float("nan")))
    return results


def _collect(
    cascade: Cascade,
    samples: Iterable[Sample],
    group: str,
    n_limit: Optional[int] = None,
):
    n_layers = cascade.n_layers
    t_raw: List[List[float]] = [[] for _ in range(n_layers)]
    t_norm: List[List[float]] = [[] for _ in range(n_layers)]
    p_raw: List[List[float]] = [[] for _ in range(n_layers)]
    p_norm: List[List[float]] = [[] for _ in range(n_layers)]
    records: List[SampleDrift] = []
    count = 0
    for s in samples:
        drift: LayerDrift = cascade.layer_drift(
            s.clean, s.shifted, s.true_label, s.pred_label_shifted
        )
        for k in range(n_layers):
            t_raw[k].append(drift.dk_true_raw[k])
            t_norm[k].append(drift.dk_true_norm[k])
            p_raw[k].append(drift.dk_pred_raw[k])
            p_norm[k].append(drift.dk_pred_norm[k])
        records.append(
            SampleDrift(
                index=s.index,
                layer_names=list(drift.layer_names),
                dk_true_raw=list(drift.dk_true_raw),
                dk_true_norm=list(drift.dk_true_norm),
                dk_pred_raw=list(drift.dk_pred_raw),
                dk_pred_norm=list(drift.dk_pred_norm),
                true_label=s.true_label,
                pred_label=s.pred_label_shifted,
                group=group,
            )
        )
        count += 1
        if n_limit is not None and count >= n_limit:
            break
    return count, t_raw, t_norm, p_raw, p_norm, records


def _profile_group(
    name: str, cascade, samples, n_limit
) -> Tuple[Optional[GroupProfile], List[SampleDrift]]:
    samples = list(samples)
    if not samples:
        return None, []
    count, t_raw, t_norm, p_raw, p_norm, records = _collect(
        cascade, samples, group=name, n_limit=n_limit
    )
    if count == 0:
        return None, []
    return (
        GroupProfile(
            name=name,
            n_samples=count,
            true_raw=_summarise(t_raw),
            true_norm=_summarise(t_norm),
            pred_raw=_summarise(p_raw),
            pred_norm=_summarise(p_norm),
            growth_true_norm=_growth(_summarise(t_norm).mean),
            ttests_true_rel=_paired_ttests_rel(t_norm),
            wilcoxon_true_norm=_wilcoxon(t_norm),
        ),
        records,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fragility_profile(
    cascade: Cascade,
    samples_incorrect: Optional[Iterable[Sample]] = None,
    samples_correct: Optional[Iterable[Sample]] = None,
    n_incorrect: Optional[int] = None,
    n_correct: Optional[int] = None,
    legacy_samples: Optional[Iterable[Tuple]] = None,
) -> FragilityProfile:
    """Build per-group attribution-drift profiles.

    Prefer the named group arguments (samples_incorrect / samples_correct).
    legacy_samples (iterable of 4-tuples) is accepted for backwards compat
    and placed in shifted_incorrect.
    """
    legacy: List[Sample] = []
    if legacy_samples is not None:
        for i, tup in enumerate(legacy_samples):
            clean, shifted, true_l, pred_l = tup
            legacy.append(
                Sample(
                    index=-1 - i,
                    clean=clean,
                    shifted=shifted,
                    true_label=int(true_l),
                    pred_label_shifted=int(pred_l),
                    shifted_correct=False,
                )
            )

    incorrect_input: List[Sample] = list(samples_incorrect) if samples_incorrect is not None else []
    incorrect_input.extend(legacy)

    incorrect_group, rec_incorrect = _profile_group(
        "shifted_incorrect", cascade, incorrect_input, n_incorrect
    )
    correct_group, rec_correct = _profile_group(
        "shifted_correct", cascade, list(samples_correct) if samples_correct is not None else [], n_correct
    )

    all_records = rec_incorrect + rec_correct
    layer_names = (
        list(cascade.layer_names)
        if all_records == []
        else list(all_records[0].layer_names)
    )

    return FragilityProfile(
        layer_names=layer_names,
        shifted_incorrect=incorrect_group,
        shifted_correct=correct_group,
        per_sample=all_records,
    )
