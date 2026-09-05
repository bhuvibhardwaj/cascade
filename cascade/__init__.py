"""
Cascade — a layer-wise attribution-drift diagnostic for CNNs under
distribution shift.

Terminology used throughout:
  * D(k) — layer-wise attribution drift. The default is D_norm(k), the
    normalized L2 distance between the GradCAM attribution of a clean image
    and a shifted image at layer k. Do not call it "spurious signal
    strength" — that is a scientific claim that remains to be established.
  * Fragility profile — aggregate D(k) distributions across samples,
    separated into the shifted-correct (survived) and shifted-incorrect
    (failed) groups. The comparison is the scientifically meaningful piece.
  * Point of no return (PNR) — a per-sample threshold crossing that is only
    meaningful when the threshold is calibrated against a null population
    (clean-vs-clean / benign comparisons). PNR must be falsifiable (i.e. a
    sample is allowed to not cross it).
"""

from .core import Cascade, LayerDrift, find_conv_layers
from .diagnose import DiagnosisResult, diagnose
from .metrics import (
    DegeneracyDecision,
    assess_null_degeneracy,
    cosine_distance,
    layer_metrics_from_maps,
    standardize_z,
)
from .pnr import PNRThresholds, calibrate_pnr_thresholds
from .report import FragilityProfile, fragility_profile
from .samples import Sample, SampleDrift
from .shift import ShiftSpec, build_shift, custom_rotation_blur, get_preset, get_preset_spec, make_shifted_dataset

__all__ = [
    "Cascade",
    "LayerDrift",
    "find_conv_layers",
    "DegeneracyDecision",
    "assess_null_degeneracy",
    "cosine_distance",
    "layer_metrics_from_maps",
    "standardize_z",
    "DiagnosisResult",
    "diagnose",
    "PNRThresholds",
    "calibrate_pnr_thresholds",
    "FragilityProfile",
    "fragility_profile",
    "Sample",
    "SampleDrift",
    "ShiftSpec",
    "build_shift",
    "custom_rotation_blur",
    "get_preset",
    "get_preset_spec",
    "make_shifted_dataset",
]

__version__ = "0.1.0"
