"""
Additional attribution-drift metrics for Experiment 1.

Existing Cascade.dk() is unchanged:
    D_raw(k)  = ||A_shift - A_clean||_2
    D_norm(k) = D_raw(k) / (||A_clean||_2 + NORM_EPSILON)

This module adds:
    D_cos(k)  = 1 - cosine_similarity(flatten(A_shift), flatten(A_clean))
    Z_*(k)    = (D_raw_shift(k) - mean D_null_*(k)) / (std D_null_*(k) + eps)

Z_* are sensitivity analyses under different *reference distributions*,
not interchangeable confirmations of one hypothesis. A degenerate reference
is recorded and yields NaN; it is never replaced by another reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .core import NORM_EPSILON

COSINE_EPSILON = 1e-8
Z_EPSILON = 1e-8
NULL_STD_MIN = 1e-8
NULL_MAXABS_MIN = 1e-6

REASON_STD = "std_below_threshold"
REASON_MAXABS = "maxabs_below_threshold"


@dataclass(frozen=True)
class DegeneracyDecision:
    degenerate: bool
    reason: Optional[str]
    std: float
    max_abs: float
    n: int
    std_min: float = NULL_STD_MIN
    maxabs_min: float = NULL_MAXABS_MIN

    def to_dict(self) -> dict:
        return asdict(self)


def cosine_distance(
    a: torch.Tensor, b: torch.Tensor, eps: float = COSINE_EPSILON
) -> float:
    """D_cos = 1 - cosine_similarity(flatten(a), flatten(b))."""
    va = a.reshape(-1).float()
    vb = b.reshape(-1).float()
    na = torch.linalg.vector_norm(va)
    nb = torch.linalg.vector_norm(vb)
    denom = (na * nb).clamp_min(eps)
    cos = torch.dot(va, vb) / denom
    cos = torch.clamp(cos, -1.0, 1.0)
    return float((1.0 - cos).item())


def d_raw_d_rel_from_maps(
    a_clean: torch.Tensor, a_shift: torch.Tensor, eps: float = NORM_EPSILON
) -> Tuple[float, float]:
    """Same arithmetic as Cascade.dk(); does not call or modify dk()."""
    diff = a_shift - a_clean
    d_raw = diff.norm().item()
    denom = a_clean.norm().item() + eps
    return float(d_raw), float(d_raw / denom)


def layer_metrics_from_maps(
    maps_clean: Sequence[torch.Tensor],
    maps_shift: Sequence[torch.Tensor],
) -> Tuple[List[float], List[float], List[float]]:
    """Per-layer (D_raw, D_rel, D_cos). D_rel uses the existing D_norm formula."""
    d_raw: List[float] = []
    d_rel: List[float] = []
    d_cos: List[float] = []
    for a_clean, a_shift in zip(maps_clean, maps_shift):
        raw, rel = d_raw_d_rel_from_maps(a_clean, a_shift)
        d_raw.append(raw)
        d_rel.append(rel)
        d_cos.append(cosine_distance(a_clean, a_shift))
    return d_raw, d_rel, d_cos


def gaussian_two_view(
    x: torch.Tensor,
    sigma: float,
    seed_a: int,
    seed_b: int,
    clamp01: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two independently noised views of the same model-input tensor."""
    g_a = torch.Generator(device="cpu").manual_seed(int(seed_a))
    g_b = torch.Generator(device="cpu").manual_seed(int(seed_b))
    x_cpu = x.detach().cpu()
    view_a = x_cpu + float(sigma) * torch.randn_like(x_cpu, generator=g_a)
    view_b = x_cpu + float(sigma) * torch.randn_like(x_cpu, generator=g_b)
    if clamp01:
        view_a = view_a.clamp(0.0, 1.0)
        view_b = view_b.clamp(0.0, 1.0)
    return view_a, view_b


def _snapshot_torch_rng_states() -> dict:
    """CPU (+ CUDA/MPS if present). Used to isolate torch.manual_seed."""
    snap: dict = {"cpu": torch.get_rng_state().clone()}
    if torch.cuda.is_available():
        snap["cuda"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    if hasattr(torch, "mps") and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
            snap["mps"] = torch.mps.get_rng_state().clone()
    return snap


def _restore_torch_rng_states(snap: dict) -> None:
    torch.set_rng_state(snap["cpu"])
    if "cuda" in snap:
        torch.cuda.set_rng_state_all(snap["cuda"])
    if "mps" in snap:
        torch.mps.set_rng_state(snap["mps"])


def apply_with_isolated_torch_seed(seed: int, fn):
    """Run ``fn()`` after ``torch.manual_seed(seed)``, then restore global RNG.

    Preserves the same draw distribution as seeding the global generator
    (what torchvision RandomCrop / RandomHorizontalFlip use) without leaving
    that generator mutated for later evaluation.
    """
    snap = _snapshot_torch_rng_states()
    try:
        torch.manual_seed(int(seed))
        return fn()
    finally:
        _restore_torch_rng_states(snap)


def assess_null_degeneracy(
    values: Sequence[float],
    std_min: float = NULL_STD_MIN,
    maxabs_min: float = NULL_MAXABS_MIN,
) -> DegeneracyDecision:
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        return DegeneracyDecision(
            degenerate=True,
            reason="empty_null",
            std=float("nan"),
            max_abs=float("nan"),
            n=0,
            std_min=std_min,
            maxabs_min=maxabs_min,
        )
    max_abs = float(np.max(np.abs(arr)))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if max_abs < maxabs_min:
        return DegeneracyDecision(
            True, REASON_MAXABS, std, max_abs, n, std_min, maxabs_min
        )
    if std < std_min:
        return DegeneracyDecision(
            True, REASON_STD, std, max_abs, n, std_min, maxabs_min
        )
    return DegeneracyDecision(False, None, std, max_abs, n, std_min, maxabs_min)


def standardize_z(
    d_shift: Sequence[float],
    null_mean: float,
    null_std: float,
    decision: DegeneracyDecision,
    eps: float = Z_EPSILON,
) -> np.ndarray:
    """Return Z for each eval sample at one layer, or NaN if this reference is degenerate.

    Does not substitute a different reference distribution.
    """
    x = np.asarray(list(d_shift), dtype=float)
    if decision.degenerate:
        return np.full_like(x, np.nan, dtype=float)
    return (x - float(null_mean)) / (float(null_std) + eps)
