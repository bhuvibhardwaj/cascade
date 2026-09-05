import pytest
import torch
import torch.nn as nn
import numpy as np

from cascade.core import Cascade, find_conv_layers, NORM_EPSILON, LayerDrift
from cascade.diagnose import diagnose, DiagnosisResult
from cascade.report import fragility_profile, GroupProfile
from cascade.samples import Sample
from cascade.shift import (
    ShiftSpec,
    build_shift,
    make_shifted_dataset,
    get_preset_spec,
    custom_rotation_blur,
)
from cascade.pnr import calibrate_pnr_thresholds


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


def _fake_pair(seed=0):
    g = torch.Generator().manual_seed(seed)
    clean = torch.rand(1, 8, 8, generator=g)
    shifted = torch.rand(1, 8, 8, generator=g)
    return clean, shifted


def test_find_conv_layers():
    model = TinyCNN()
    layers = find_conv_layers(model)
    assert len(layers) == 2

    layers_capped = find_conv_layers(model, max_layers=1)
    assert len(layers_capped) == 1


# ==========================================================================
# Scientific correctness tests
# ==========================================================================

def test_identical_inputs_produce_near_zero_dk():
    """Same image → attribution drift should be numerically near zero.

    If this fails, something is introducing spurious state into the GradCAM
    computation (e.g. unseeded randomness, dropout still on, etc.).
    """
    torch.manual_seed(0)
    model = TinyCNN()
    cascade = Cascade(model)
    g = torch.Generator().manual_seed(1)
    x = torch.rand(1, 8, 8, generator=g)
    dk_raw, dk_norm = cascade.dk(x, x, target_class=0)
    assert all(np.isfinite(v) for v in dk_raw)
    assert all(np.isfinite(v) for v in dk_norm)
    # Drift of an image vs itself should be essentially zero — we allow a
    # generous tolerance for any float nondeterminism.
    assert max(dk_raw) < 1e-4, f"Self-vs-self D(k) non-zero: max={max(dk_raw)}"


def test_dk_returns_raw_and_normalized():
    model = TinyCNN()
    cascade = Cascade(model)
    clean, shifted = _fake_pair(0)
    dk_raw, dk_norm = cascade.dk(clean, shifted, target_class=0)
    assert len(dk_raw) == cascade.n_layers == 2
    assert len(dk_norm) == 2
    assert all(v >= 0 for v in dk_raw)
    assert all(v >= 0 for v in dk_norm)
    # D_norm should be roughly D_raw / (||A_clean|| + eps) — so normally <= D_raw
    for r, n in zip(dk_raw, dk_norm):
        assert r == 0 or np.isfinite(n)


def test_layer_drift_has_raw_and_norm():
    model = TinyCNN()
    cascade = Cascade(model)
    clean, shifted = _fake_pair(0)
    drift: LayerDrift = cascade.layer_drift(clean, shifted, 0, 1)
    assert len(drift.dk_true_raw) == 2
    assert len(drift.dk_true_norm) == 2
    assert len(drift.dk_pred_raw) == 2
    assert len(drift.dk_pred_norm) == 2
    # Property-aliases default to normalized values
    assert drift.dk_true is drift.dk_true_norm
    assert drift.dk_pred is drift.dk_pred_norm


# ==========================================================================
# Shift determinism
# ==========================================================================

def test_shift_is_deterministic():
    """Same input + same spec MUST produce byte-identical output every call."""
    spec = ShiftSpec(degrees=30.0, blur_kernel=3, blur_sigma=1.0)
    fn = build_shift(spec)
    torch.manual_seed(0)
    x = torch.rand(1, 8, 8)
    y1 = fn(x)
    y2 = fn(x)
    assert torch.equal(y1, y2), "build_shift() produced different outputs on identical inputs"


def test_custom_rotation_blur_matches_build_shift():
    spec = ShiftSpec(degrees=45.0, blur_kernel=3, blur_sigma=1.2)
    f1 = build_shift(spec)
    f2 = custom_rotation_blur(45.0, 3, blur_sigma=1.2)
    torch.manual_seed(0)
    x = torch.rand(1, 16, 16)
    assert torch.allclose(f1(x), f2(x), atol=1e-6)


def test_shifted_dataset_access_idempotent():
    """Accessing shifted dataset at same index twice must yield same tensor."""
    torch.manual_seed(0)
    ds = [
        (torch.rand(1, 6, 6), i) for i in range(4)
    ]

    class FakeDS:
        def __init__(self, data): self._d = data
        def __len__(self): return len(self._d)
        def __getitem__(self, i): return self._d[i]

    sds = make_shifted_dataset(FakeDS(ds), get_preset_spec("mild"))
    x0a, _ = sds[0]
    x0b, _ = sds[0]
    assert torch.equal(x0a, x0b)


# ==========================================================================
# Sample/index pairing audit
# ==========================================================================

def test_sample_index_tracking_and_tuple_roundtrip():
    clean = torch.rand(1, 4, 4)
    shifted = torch.rand(1, 4, 4)
    s = Sample(
        index=417,
        clean=clean,
        shifted=shifted,
        true_label=2,
        pred_label_shifted=7,
        shifted_correct=False,
    )
    assert s.index == 417
    c, sh, t, p = s.as_tuple()
    assert t == 2
    assert p == 7
    assert torch.equal(c, clean)
    assert torch.equal(sh, shifted)


# ==========================================================================
# Paired statistics + four-group profile
# ==========================================================================

def test_fragility_profile_supports_shift_correct_and_incorrect():
    model = TinyCNN()
    cascade = Cascade(model)

    def make_samples(group: str, n, seed_base):
        out = []
        for j in range(n):
            clean, shifted = _fake_pair(seed_base + j)
            out.append(Sample(
                index=seed_base + j,
                clean=clean,
                shifted=shifted,
                true_label=0,
                pred_label_shifted=1 if group == "incorrect" else 0,
                shifted_correct=(group == "correct"),
            ))
        return out

    inc = make_samples("incorrect", n=4, seed_base=0)
    cor = make_samples("correct", n=4, seed_base=1000)
    profile = fragility_profile(
        cascade,
        samples_incorrect=inc,
        samples_correct=cor,
    )
    assert isinstance(profile.shifted_incorrect, GroupProfile)
    assert isinstance(profile.shifted_correct, GroupProfile)
    assert profile.shifted_incorrect.n_samples == 4
    assert profile.shifted_correct.n_samples == 4
    # Paired tests exist for both groups
    assert len(profile.shifted_incorrect.ttests_true_rel) == 1
    assert len(profile.shifted_incorrect.wilcoxon_true_norm) == 1
    # Per-sample records exist with group tag
    groups_seen = sorted(set(r.group for r in profile.per_sample))
    assert groups_seen == ["shifted_correct", "shifted_incorrect"]
    # All records have dataset indices
    assert all(r.index >= 0 for r in profile.per_sample)


def test_legacy_4tuple_path_still_works():
    """Existing code that passes 4-tuples should continue to work (backward compat)."""
    model = TinyCNN()
    cascade = Cascade(model)
    legacy = [(_fake_pair(i) + (0, 1)) for i in range(3)]
    profile = fragility_profile(cascade, legacy_samples=legacy)
    assert profile.shifted_incorrect is not None
    assert profile.shifted_incorrect.n_samples == 3


# ==========================================================================
# PNR falsifiability
# ==========================================================================

def test_pnr_can_be_none_for_benign_pairs():
    """Identical clean/clean samples vs calibrated threshold → PNR should
    generally be None (falsifiable: not every sample must have a PNR)."""
    torch.manual_seed(0)
    model = TinyCNN()
    cascade = Cascade(model)
    # Null calibration set
    g = torch.Generator().manual_seed(7)
    null_pairs = []
    for i in range(16):
        x = torch.rand(1, 8, 8, generator=g)
        null_pairs.append((x, x, i % 3))
    thresh = calibrate_pnr_thresholds(cascade, null_pairs, quantile=0.95)
    # Test: feed the same null pair
    x = torch.rand(1, 8, 8, generator=torch.Generator().manual_seed(1))
    res = diagnose(cascade, x, x, 0, 0, threshold=thresh)
    # Self-vs-self should NOT exceed 95th percentile of self-vs-self null;
    # so PNR should be None (falsified)
    assert res.point_of_no_return is None, (
        "Self-vs-self exceeded PNR threshold — "
        "calibration must be flawed or GradCAM has spurious state."
    )


# ==========================================================================
# Legacy surface tests (still valid)
# ==========================================================================

def test_dk_returns_one_value_per_layer():
    model = TinyCNN()
    cascade = Cascade(model)
    clean, shifted = _fake_pair()
    dk_raw, dk_norm = cascade.dk(clean, shifted, target_class=0)
    assert len(dk_raw) == cascade.n_layers == 2
    assert all(isinstance(v, float) for v in dk_raw)
    assert all(v >= 0 for v in dk_raw)


def test_layer_drift_dual_label():
    model = TinyCNN()
    cascade = Cascade(model)
    clean, shifted = _fake_pair()
    drift = cascade.layer_drift(clean, shifted, true_label=0, pred_label=1)
    assert len(drift.dk_true_raw) == len(drift.dk_pred_raw) == cascade.n_layers
    assert len(drift.dk_true_norm) == len(drift.dk_pred_norm) == cascade.n_layers


def test_diagnose_produces_verdict():
    model = TinyCNN()
    cascade = Cascade(model)
    clean, shifted = _fake_pair()
    result = diagnose(cascade, clean, shifted, true_label=0, pred_label=1)
    assert result.verdict in {"stable", "unstable"}
    assert len(result.dk_true) == cascade.n_layers


def test_fragility_profile_over_multiple_samples():
    model = TinyCNN()
    cascade = Cascade(model)
    legacy = [(_fake_pair(i) + (0, 1)) for i in range(5)]
    profile = fragility_profile(cascade, legacy_samples=legacy)
    assert profile.shifted_incorrect is not None
    assert profile.shifted_incorrect.n_samples == 5
    assert len(profile.shifted_incorrect.true_norm.mean) == cascade.n_layers
    assert len(profile.shifted_incorrect.growth_true_norm) == cascade.n_layers - 1
    assert len(profile.shifted_incorrect.ttests_true_rel) == cascade.n_layers - 1
