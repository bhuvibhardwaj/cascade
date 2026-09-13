"""
Core Cascade engine.

Measures layer-wise attribution drift D(k) via GradCAM:

    D_raw(k)  = || GradCAM_shifted(k) - GradCAM_clean(k) ||
    D_norm(k) = D_raw(k) / ( || GradCAM_clean(k) || + epsilon )

D_raw is an L2 distance in attribution space; D_norm normalizes by the
magnitude of the clean attribution so layers with naturally larger
representations are not over-weighted. Report both — D_norm is the safer
choice for cross-layer comparison and for defining drift thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

try:
    from captum.attr import LayerGradCam
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "cascade requires captum. Install it with `pip install captum`."
    ) from exc


NORM_EPSILON = 1e-8


def find_conv_layers(
    model: nn.Module, max_layers: Optional[int] = None
) -> List[Tuple[str, nn.Module]]:
    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    if not layers:
        raise ValueError(
            "No nn.Conv2d layers found in this model. "
            "Cascade currently only supports CNNs; pass `layers` explicitly "
            "to Cascade(...) if your architecture uses a different conv type."
        )
    if max_layers is not None and max_layers < len(layers):
        idx = torch.linspace(0, len(layers) - 1, max_layers).round().long().tolist()
        idx = sorted(set(idx))
        layers = [layers[i] for i in idx]
    return layers


@dataclass
class LayerDrift:
    """Layer-wise attribution drift for a single (clean, shifted) pair.

    dk_*_raw  = raw L2 attribution-drift per layer
    dk_*_norm = normalized drift: ||Δ|| / (||clean|| + ε)
    """

    layer_names: List[str]
    dk_true_raw: List[float]
    dk_true_norm: List[float]
    dk_pred_raw: List[float]
    dk_pred_norm: List[float]

    @property
    def dk_true(self) -> List[float]:
        """Alias for D_norm (normalized) — use .dk_true_raw for the raw norm."""
        return self.dk_true_norm

    @property
    def dk_pred(self) -> List[float]:
        return self.dk_pred_norm


class Cascade:
    """Wrap a trained CNN and compute layer-wise attribution drift D(k).

    Usage:
        cascade = Cascade(model, device="cuda")
        drift = cascade.layer_drift(clean_img, shifted_img, true_label, pred_label)
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[Sequence[Tuple[str, nn.Module]]] = None,
        max_layers: Optional[int] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.model = model.to(device).eval()
        self.layers = list(layers) if layers is not None else find_conv_layers(
            self.model, max_layers=max_layers
        )
        self.layer_names = [name for name, _ in self.layers]
        self.gradcam_layers = [
            LayerGradCam(self.model, module) for _, module in self.layers
        ]

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def _attribute(
        self, image: torch.Tensor, target_class: int
    ) -> List[torch.Tensor]:
        img = image
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device).requires_grad_(True)
        return [gc.attribute(img, target=target_class) for gc in self.gradcam_layers]

    def attribution_maps(
        self, image: torch.Tensor, target_class: int
    ) -> List[torch.Tensor]:
        """Return GradCAM maps using the same path as ``dk()``.

        Signature: attribution_maps(image, target_class) -> List[Tensor]
        Does not modify ``dk()`` or the D_raw / D_norm formulas.
        """
        return self._attribute(image, target_class)

    def dk(
        self,
        clean_image: torch.Tensor,
        shifted_image: torch.Tensor,
        target_class: int,
        return_maps: bool = False,
    ) -> Union[
        Tuple[List[float], List[float]],
        Tuple[List[float], List[float], List[torch.Tensor], List[torch.Tensor]],
    ]:
        """Return (D_raw, D_norm) per layer for a given target_class.

        D_raw(k)  = || G_k^shift - G_k^clean ||_2   (computed, then maps kept)
        D_norm(k) = D_raw(k) / ( || G_k^clean ||_2 + eps )

        The GradCAM tensors A_k(clean) and A_k(shifted) stay in scope through
        the .norm() calls. Default return is still only the scalars (callers
        unchanged). Pass return_maps=True to also get those two lists, so a
        later metric (e.g. cosine distance) can reuse one forward without
        another GradCAM pass.
        """
        attrs_clean = self._attribute(clean_image, target_class)
        attrs_shifted = self._attribute(shifted_image, target_class)
        dk_raw: List[float] = []
        dk_norm: List[float] = []
        for a_clean, a_shift in zip(attrs_clean, attrs_shifted):
            diff = a_shift - a_clean
            d_raw = diff.norm().item()
            denom = a_clean.norm().item() + NORM_EPSILON
            dk_raw.append(float(d_raw))
            dk_norm.append(float(d_raw / denom))
        if return_maps:
            return dk_raw, dk_norm, attrs_clean, attrs_shifted
        return dk_raw, dk_norm

    def layer_drift(
        self,
        clean_image: torch.Tensor,
        shifted_image: torch.Tensor,
        true_label: int,
        pred_label: int,
    ) -> LayerDrift:
        t_raw, t_norm = self.dk(clean_image, shifted_image, true_label)
        p_raw, p_norm = self.dk(clean_image, shifted_image, pred_label)
        return LayerDrift(
            layer_names=self.layer_names,
            dk_true_raw=t_raw,
            dk_true_norm=t_norm,
            dk_pred_raw=p_raw,
            dk_pred_norm=p_norm,
        )
