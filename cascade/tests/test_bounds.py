"""Statistical layer for PNR thresholds (math-backing Sections 3–5)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy import stats

from cascade.bounds import (
    FARResult,
    benjamini_hochberg,
    benjamini_yekutieli,
    calibrate_joint_from_trajectories,
    calibrate_thresholds_bonferroni,
    calibrate_thresholds_joint,
    dkw_epsilon,
    dkw_epsilon_simultaneous,
    far_with_ci,
    mann_whitney_layer_test,
    thresholds_at_quantile,
    trajectory_false_alarm_rate,
    wilson_proportion_interval,
)
from cascade.core import Cascade
from cascade.diagnose import diagnose
from cascade.pnr import PNRThresholds, calibrate_pnr_thresholds, collect_dk_trajectories
from cascade.significance import layer_significance_table


class TinyCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


# Wikipedia / BH textbook example (m=15, q=0.05): reject the four smallest.
BH_TEXTBOOK_P = [
    0.0001,
    0.0004,
    0.0019,
    0.0095,
    0.0201,
    0.0278,
    0.0298,
    0.0344,
    0.0459,
    0.3240,
    0.4262,
    0.5719,
    0.6528,
    0.7590,
    1.000,
]


def test_dkw_matches_math_backing_worked_example():
    """Section 3 worked example: n=500, L=8, δ=0.05."""
    eps = dkw_epsilon(500, delta=0.05)
    eps_sim = dkw_epsilon_simultaneous(500, 8, delta=0.05)
    assert eps == pytest.approx(0.0607, abs=5e-4)
    assert eps_sim == pytest.approx(0.0759, abs=5e-4)
    assert eps_sim > eps


def test_dkw_monte_carlo_coverage_of_quantile():
    """DKW is a uniform-over-x bound; |F(θ̂_q) − q| ≤ ε should hold at rate ≥ 1−δ.

    True law: Exponential(1). Event is weaker than the full DKW event, so
    empirical coverage may exceed 1−δ; we only require it not fall below
    1−δ by more than Monte Carlo slack.
    """
    rng = np.random.default_rng(0)
    n = 200
    q = 0.95
    delta = 0.05
    eps = dkw_epsilon(n, delta=delta)
    n_trials = 400
    violations = 0
    for _ in range(n_trials):
        sample = rng.exponential(scale=1.0, size=n)
        theta = float(np.quantile(sample, q))
        f_theta = 1.0 - np.exp(-theta)
        if abs(f_theta - q) > eps:
            violations += 1
    rate = violations / n_trials
    # Monte Carlo SE of a 0.05 binomial is ~0.011; allow 3 SE slack.
    assert rate <= delta + 0.04, f"DKW coverage failed: violation rate={rate:.3f}"


def test_mwu_matches_scipy_on_stochastically_larger_shifted():
    rng = np.random.default_rng(1)
    null = rng.normal(0.0, 1.0, size=40)
    shifted = rng.normal(0.6, 1.0, size=40)
    ours = mann_whitney_layer_test(null, shifted)
    sc = stats.mannwhitneyu(shifted, null, alternative="greater", method="asymptotic")
    assert ours.method == "normal_approximation"
    assert ours.u_statistic == pytest.approx(float(sc.statistic), rel=0, abs=1e-9)
    assert ours.p_value == pytest.approx(float(sc.pvalue), rel=1e-10, abs=1e-10)
    assert ours.z_score == pytest.approx(float(sc.zstatistic), rel=1e-10, abs=1e-10)
    assert ours.p_value < 0.05


def test_mwu_small_sample_uses_scipy_auto():
    null = [1.0, 2.0, 3.0, 4.0]
    shifted = [3.5, 4.5, 5.5]
    ours = mann_whitney_layer_test(null, shifted)
    sc = stats.mannwhitneyu(shifted, null, alternative="greater", method="auto")
    assert ours.method == "scipy_auto"
    assert ours.p_value == pytest.approx(float(sc.pvalue), rel=1e-12, abs=1e-12)


def test_mwu_null_vs_null_pvalues_are_roughly_uniform():
    rng = np.random.default_rng(2)
    pvals = []
    for _ in range(250):
        a = rng.normal(size=30)
        b = rng.normal(size=30)
        pvals.append(mann_whitney_layer_test(a, b).p_value)
    pvals = np.asarray(pvals)
    # KS against Uniform(0,1); should not reject at 0.01 under a true null.
    _, ks_p = stats.kstest(pvals, "uniform")
    assert ks_p > 0.01, f"null-vs-null p-values not uniform (KS p={ks_p:.4f})"
    frac = float(np.mean(pvals < 0.05))
    assert 0.015 <= frac <= 0.10, f"type-I rate {frac:.3f} far from 0.05"


def test_bh_textbook_example_rejects_first_four():
    result = benjamini_hochberg(BH_TEXTBOOK_P, q=0.05)
    assert result.cutoff_rank == 4
    assert result.rejected == [True, True, True, True] + [False] * 11
    assert all(result.p_adjusted[i] <= result.p_adjusted[i + 1] + 1e-12 for i in range(3))


def test_by_is_more_conservative_than_bh():
    bh = benjamini_hochberg(BH_TEXTBOOK_P, q=0.05)
    by = benjamini_yekutieli(BH_TEXTBOOK_P, q=0.05)
    assert by.cutoff_rank <= bh.cutoff_rank
    assert sum(by.rejected) <= sum(bh.rejected)
    for a, b in zip(by.p_adjusted, bh.p_adjusted):
        assert a + 1e-12 >= b


def test_naive_independent_far_near_doc_figure():
    """L=8 independent layers, naive 95th → ~33.7% union FAR (Section 3.2)."""
    rng = np.random.default_rng(3)
    n, L = 5000, 8
    traj = rng.normal(size=(n, L))
    theta = thresholds_at_quantile(traj, 0.95)
    far = trajectory_false_alarm_rate(traj, theta)
    # In-sample quantile is slightly conservative vs the population 0.3366.
    assert 0.28 <= far.rate <= 0.38, f"naive FAR={far.rate:.4f}, expected ~0.337"
    assert far.n == n
    assert far.n_triggered + (n - far.n_triggered) == n


def test_bonferroni_independent_far_near_target():
    rng = np.random.default_rng(4)
    n, L = 5000, 8
    target = 0.05
    traj = rng.normal(size=(n, L))
    q = 1.0 - target / L
    theta = thresholds_at_quantile(traj, q)
    far = trajectory_false_alarm_rate(traj, theta)
    assert abs(far.rate - target) < 0.02, f"Bonferroni FAR={far.rate:.4f}"


def test_joint_hits_target_and_is_tighter_than_bonferroni_when_correlated():
    rng = np.random.default_rng(5)
    n, L = 4000, 8
    target = 0.05
    common = rng.normal(size=(n, 1))
    noise = rng.normal(size=(n, L))
    traj = common + 0.15 * noise  # strong positive layer correlation

    theta_b = thresholds_at_quantile(traj, 1.0 - target / L)
    far_b = trajectory_false_alarm_rate(traj, theta_b)
    theta_c, q_c, far_c = calibrate_joint_from_trajectories(traj, target_fdr=target)

    assert far_c <= target + 0.005
    assert abs(far_c - target) < 0.02
    # Correlation makes Bonferroni conservative; joint should be more sensitive.
    assert far_b.rate < far_c + 0.005
    assert float(np.mean(theta_c)) <= float(np.mean(theta_b)) + 1e-12
    # Multiplier on the naive 95th-percentile template; correlated layers
    # should not need a large inflation.
    assert q_c <= 1.15


def test_naive_calibrate_still_default_and_diagnose_surfaces_method():
    torch.manual_seed(0)
    cascade = Cascade(TinyCNN())
    g = torch.Generator().manual_seed(7)
    null_pairs = []
    for i in range(12):
        x = torch.rand(1, 8, 8, generator=g)
        y = x + 0.05 * torch.rand(1, 8, 8, generator=g)
        null_pairs.append((x, y, i % 3))
    thresh = calibrate_pnr_thresholds(cascade, null_pairs, quantile=0.95)
    assert thresh.calibration_method == "naive"
    assert thresh.epsilon is not None and thresh.epsilon_simultaneous is not None
    assert thresh.epsilon_simultaneous >= thresh.epsilon
    assert thresh.n_pairs == 12
    x, y, _ = null_pairs[0]
    res = diagnose(cascade, x, y, 0, 0, threshold=thresh)
    text = res.summary()
    assert "naive" in text
    assert "ε" in text or "epsilon" in text.lower() or "DKW" in text
    assert res.calibration_method == "naive"


def test_bonferroni_and_joint_calibrators_return_pnrthresholds():
    torch.manual_seed(1)
    cascade = Cascade(TinyCNN())
    g = torch.Generator().manual_seed(11)
    pairs = []
    for i in range(16):
        a = torch.rand(1, 8, 8, generator=g)
        b = a + 0.08 * torch.rand(1, 8, 8, generator=g)
        pairs.append((a, b, 0))
    b = calibrate_thresholds_bonferroni(cascade, pairs, target_fdr=0.05)
    with pytest.warns(UserWarning, match="tune_pairs"):
        c = calibrate_thresholds_joint(cascade, pairs, target_fdr=0.05)
    assert isinstance(b, PNRThresholds) and isinstance(c, PNRThresholds)
    assert b.calibration_method == "bonferroni"
    assert c.calibration_method == "joint"
    assert len(b.values) == cascade.n_layers == len(c.values)
    assert b.quantile == pytest.approx(1.0 - 0.05 / cascade.n_layers)


def test_layer_significance_table_detects_shifted_increase():
    torch.manual_seed(2)
    cascade = Cascade(TinyCNN())
    g = torch.Generator().manual_seed(13)
    null_pairs = []
    shifted_pairs = []
    for i in range(16):
        x = torch.rand(1, 8, 8, generator=g)
        null_pairs.append((x, x + 0.02 * torch.rand(1, 8, 8, generator=g), 0))
        shifted_pairs.append((x, x + 0.5 * torch.rand(1, 8, 8, generator=g), 0))
    table = layer_significance_table(cascade, null_pairs, shifted_pairs, q=0.05)
    assert table.n_null == 16 and table.n_shifted == 16
    assert len(table.rows) == cascade.n_layers
    assert table.fdr_method == "bh"
    # Large synthetic shift should be detectable at FDR 5% in at least one layer.
    assert any(r.rejected for r in table.rows)


def test_wilson_ci_on_small_n_is_wide():
    """1/8 is not '12.5%' in any decision-relevant sense — the CI spans tens of points."""
    lo, hi = wilson_proportion_interval(1, 8, alpha=0.05)
    rate, lo2, hi2 = far_with_ci(1, 8, alpha=0.05)
    assert rate == pytest.approx(0.125)
    assert (lo, hi) == (lo2, hi2)
    # statsmodels Wilson for 1/8 at 95% is about [2.2%, 47.1%].
    assert lo == pytest.approx(0.022, abs=0.005)
    assert hi == pytest.approx(0.471, abs=0.005)
    formatted = FARResult(n_triggered=1, n=8, rate=0.125).format_ci()
    assert "1/8 = 12.5%" in formatted
    assert "2.2%" in formatted and "47.1%" in formatted


def test_joint_respects_external_template_on_tune_fold():
    rng = np.random.default_rng(6)
    calib = rng.normal(size=(400, 8))
    tune = rng.normal(size=(400, 8))
    base = thresholds_at_quantile(calib, 0.95)
    thr, m, far_tune = calibrate_joint_from_trajectories(
        tune, target_fdr=0.05, base_thresholds=base
    )
    assert m >= 1.0
    assert far_tune <= 0.05 + 1e-9
    assert thr.shape == (8,)


def test_same_fold_joint_overfits_relative_to_holdout():
    """Same-fold Option C matches target in-sample; that FAR does not transfer."""
    rng = np.random.default_rng(8)
    fit = rng.normal(size=(80, 8))
    holdout = rng.normal(size=(4000, 8))
    thr, _m, far_fit = calibrate_joint_from_trajectories(fit, target_fdr=0.05)
    far_hold = trajectory_false_alarm_rate(holdout, thr)
    assert far_fit <= 0.05 + 1e-9
    assert far_hold.rate > far_fit
    assert far_hold.rate > 0.05


def test_calibrate_thresholds_joint_tune_pairs_matches_lower_level():
    torch.manual_seed(3)
    cascade = Cascade(TinyCNN())
    g = torch.Generator().manual_seed(17)
    calib, tune = [], []
    for _i in range(12):
        a = torch.rand(1, 8, 8, generator=g)
        b = a + 0.08 * torch.rand(1, 8, 8, generator=g)
        calib.append((a, b, 0))
    for _i in range(12):
        a = torch.rand(1, 8, 8, generator=g)
        b = a + 0.08 * torch.rand(1, 8, 8, generator=g)
        tune.append((a, b, 0))

    calib_traj = collect_dk_trajectories(cascade, calib)
    tune_traj = collect_dk_trajectories(cascade, tune)
    base = thresholds_at_quantile(calib_traj, 0.95)
    expected, _m, far_tune = calibrate_joint_from_trajectories(
        tune_traj, target_fdr=0.05, base_thresholds=base
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = calibrate_thresholds_joint(
            cascade, calib, target_fdr=0.05, tune_pairs=tune
        )
    assert not any("tune_pairs" in str(w.message) for w in caught)
    assert got.calibration_method == "joint"
    assert got.n_pairs == 12
    assert got.values == [float(v) for v in expected]
    assert got.measured_false_alarm_rate == pytest.approx(far_tune)

    # Same object for both folds is still same-fold overfitting.
    with pytest.warns(UserWarning, match="tune_pairs"):
        calibrate_thresholds_joint(
            cascade, calib, target_fdr=0.05, tune_pairs=calib
        )
