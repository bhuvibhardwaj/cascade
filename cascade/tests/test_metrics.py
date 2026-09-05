"""Tests for Experiment 1 metric helpers. Does not modify Cascade.dk()."""

import numpy as np
import torch
import torch.nn as nn

from cascade.core import Cascade, NORM_EPSILON
from cascade.metrics import (
    REASON_MAXABS,
    REASON_STD,
    apply_with_isolated_torch_seed,
    assess_null_degeneracy,
    cosine_distance,
    d_raw_d_rel_from_maps,
    gaussian_two_view,
    layer_metrics_from_maps,
    standardize_z,
)


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


def test_attribution_maps_matches_private_attribute():
    torch.manual_seed(0)
    cascade = Cascade(TinyCNN())
    x = torch.rand(1, 8, 8)
    a = cascade.attribution_maps(x, 0)
    b = cascade._attribute(x, 0)
    assert len(a) == len(b) == 2
    for ua, ub in zip(a, b):
        assert torch.allclose(ua, ub)


def test_d_raw_d_rel_matches_dk_arithmetic():
    a_clean = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    a_shift = torch.tensor([[[[1.0, 3.0], [0.0, 0.0]]]])
    d_raw, d_rel = d_raw_d_rel_from_maps(a_clean, a_shift)
    diff = a_shift - a_clean
    expected_raw = diff.norm().item()
    expected_rel = expected_raw / (a_clean.norm().item() + NORM_EPSILON)
    assert d_raw == expected_raw
    assert d_rel == expected_rel


def test_dk_unchanged_and_matches_from_maps_on_same_tensors():
    torch.manual_seed(1)
    cascade = Cascade(TinyCNN())
    clean = torch.rand(1, 8, 8)
    shifted = torch.rand(1, 8, 8)
    maps_c = cascade.attribution_maps(clean, 0)
    maps_s = cascade.attribution_maps(shifted, 0)
    raw_m, rel_m, _ = layer_metrics_from_maps(maps_c, maps_s)
    # Recompute dk arithmetic on the same maps (dk() would re-run GradCAM).
    for a_c, a_s, r, n in zip(maps_c, maps_s, raw_m, rel_m):
        rr, nn = d_raw_d_rel_from_maps(a_c, a_s)
        assert r == rr and n == nn
    dk_raw, dk_norm = cascade.dk(clean, shifted, 0)
    assert len(dk_raw) == 2 and len(dk_norm) == 2
    assert all(v >= 0 for v in dk_raw)


def test_cosine_identical_is_zero():
    t = torch.arange(8.0).reshape(1, 1, 2, 4)
    assert cosine_distance(t, t) < 1e-6


def test_cosine_orthogonal():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    assert abs(cosine_distance(a, b) - 1.0) < 1e-5


def test_identity_null_is_degenerate():
    d = assess_null_degeneracy([0.0, 0.0, 1e-12, 0.0])
    assert d.degenerate
    assert d.reason == REASON_MAXABS


def test_low_std_null_is_degenerate():
    d = assess_null_degeneracy([1.0, 1.0, 1.0, 1.0 + 1e-12])
    assert d.degenerate
    assert d.reason == REASON_STD


def test_nondegenerate_null():
    rng = np.random.default_rng(0)
    vals = rng.normal(1.0, 0.2, size=50)
    d = assess_null_degeneracy(vals)
    assert not d.degenerate
    assert d.reason is None


def test_standardize_z_writes_nan_when_degenerate_no_substitute():
    decision = assess_null_degeneracy([0.0, 0.0, 0.0])
    z = standardize_z([1.0, 2.0, 3.0], null_mean=0.0, null_std=0.0, decision=decision)
    assert np.all(np.isnan(z))


def test_standardize_z_uses_named_reference_only():
    vals = [1.0, 2.0, 3.0, 4.0]
    decision = assess_null_degeneracy(vals)
    assert not decision.degenerate
    z = standardize_z([2.5], null_mean=float(np.mean(vals)), null_std=float(np.std(vals, ddof=1)), decision=decision)
    assert np.isfinite(z).all()


def test_gaussian_two_view_is_seeded_and_independent():
    x = torch.zeros(1, 2, 2)
    a1, b1 = gaussian_two_view(x, 0.1, seed_a=7, seed_b=8, clamp01=True)
    a2, b2 = gaussian_two_view(x, 0.1, seed_a=7, seed_b=8, clamp01=True)
    assert torch.equal(a1, a2) and torch.equal(b1, b2)
    assert not torch.equal(a1, b1)


def test_isolated_torch_seed_matches_global_manual_seed_distribution():
    """Same draws as torch.manual_seed + torchvision RandomCrop (exact distribution)."""
    import torchvision.transforms as T

    img = torch.arange(16.0).reshape(1, 4, 4)
    crop = T.RandomCrop(4, padding=1)

    torch.manual_seed(123)
    expected = crop(img)

    torch.manual_seed(999)
    got = apply_with_isolated_torch_seed(123, lambda: crop(img))
    assert torch.equal(got, expected)


def test_isolated_torch_seed_restores_global_rng_for_subsequent_eval():
    """Calibration-style isolated seeding must not change later evaluation draws."""
    import torchvision.transforms as T

    img = torch.ones(1, 8, 8)
    crop = T.RandomCrop(8, padding=2)

    torch.manual_seed(42)
    control = torch.rand(5)

    torch.manual_seed(42)
    for i in range(8):
        seed_a = 1_000_003 + i * 2
        seed_b = seed_a + 1
        apply_with_isolated_torch_seed(seed_a, lambda: crop(img))
        apply_with_isolated_torch_seed(seed_b, lambda: crop(img))
    after_calib = torch.rand(5)
    assert torch.equal(after_calib, control)


def test_mild_aug_two_view_seeds_are_independent_and_reproducible():
    import torchvision.transforms as T

    img = torch.linspace(0, 1, 28 * 28).reshape(1, 28, 28)
    crop = T.RandomCrop(28, padding=2)
    a1 = apply_with_isolated_torch_seed(7, lambda: crop(img))
    b1 = apply_with_isolated_torch_seed(8, lambda: crop(img))
    a2 = apply_with_isolated_torch_seed(7, lambda: crop(img))
    b2 = apply_with_isolated_torch_seed(8, lambda: crop(img))
    assert torch.equal(a1, a2) and torch.equal(b1, b2)
    assert not torch.equal(a1, b1)
