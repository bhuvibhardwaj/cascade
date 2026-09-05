"""
Deterministic shift-synthesis presets.

Cascade's core engine only requires (clean, shifted) image pairs — it does
not depend on this module. These presets exist as a convenience wrapper that
guarantees the shift is reproducible:

    x_i^{shift} = T(x_i)   (identical every time

NOT:
    x_i^{shift} = T_{theta_i}(x_i)  (theta_i changes every access)

This is critical for D(k) measurements: a stochastic transform applied at dataset access
time would silently destroy the clean/shifted correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple, Union

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


@dataclass(frozen=True)
class ShiftSpec:
    """Fully specified deterministic shift parameters."""

    degrees: float = 0.0
    blur_kernel: int = 0
    blur_sigma: float = 0.0
    noise_std: float = 0.0
    brightness: float = 1.0
    contrast: float = 1.0

    def label(self) -> str:
        parts: List[str] = []
        if self.degrees != 0.0:
            parts.append(f"rot{self.degrees:g}")
        if self.blur_kernel > 0:
            parts.append(f"blur{self.blur_kernel}s{self.blur_sigma:g}")
        if self.noise_std > 0.0:
            parts.append(f"noise{self.noise_std:g}")
        if self.brightness != 1.0:
            parts.append(f"bri{self.brightness:g}")
        if self.contrast != 1.0:
            parts.append(f"con{self.contrast:g}")
        return "-".join(parts) if parts else "none"


def build_shift(spec: ShiftSpec) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a pure function: tensor (C,H,W) -> tensor that applies *spec* deterministically.

    Every call with the same input tensor and spec produces byte-identical output.
    """

    def _apply(x: torch.Tensor) -> torch.Tensor:
        y = x
        if spec.degrees != 0.0:
            y = TF.rotate(y, float(spec.degrees), interpolation=TF.InterpolationMode.BILINEAR)
        if spec.blur_kernel > 0:
            if spec.blur_sigma <= 0.0:
                raise ValueError(
                    "blur_sigma must be > 0 when blur_kernel > 0."
                )
            y = TF.gaussian_blur(
                y, kernel_size=[int(spec.blur_kernel), int(spec.blur_kernel)],
                sigma=[float(spec.blur_sigma), float(spec.blur_sigma)],
            )
        if spec.brightness != 1.0:
            y = TF.adjust_brightness(y, float(spec.brightness))
        if spec.contrast != 1.0:
            y = TF.adjust_contrast(y, float(spec.contrast))
        if spec.noise_std > 0.0:
            g = torch.Generator()
            g.manual_seed(0)
            noise = torch.randn_like(y) * float(spec.noise_std)
            y = y + noise
            y = y.clamp(0.0, 1.0)
        return y

    return _apply


def make_shifted_dataset(
    base_dataset,
    spec: ShiftSpec,
    to_tensor: bool = True,
):
    """Wrap a torchvision dataset, applying a deterministic shift per access.

    base_dataset[i] is expected to return (PIL_image, label) OR
    (tensor_image, label); this wrapper applies the spec and returns
    (shifted_tensor, label).

    The shifted output is identical every time index `i` is requested — the shift is a
    pure function of the base sample and spec.
    """
    return _ShiftedDataset(base_dataset=base_dataset, spec=spec, to_tensor=to_tensor)


class _ShiftedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, spec: ShiftSpec, to_tensor: bool):
        self.base = base_dataset
        self.spec = spec
        self._shift = build_shift(spec)
        self._force_to_tensor = to_tensor
        self._to_tensor_tf = T.ToTensor()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, label = self.base[idx]
        if torch.is_tensor(x):
            t = x
        elif self._force_to_tensor:
            t = self._to_tensor_tf(x)
        else:
            t = x
        return self._shift(t), label


# ---------------------------------------------------------------------------
# Named presets — kept for backwards compatibility, but now pure-function based.
# ---------------------------------------------------------------------------

MILD_SPEC = ShiftSpec(degrees=30.0, blur_kernel=3, blur_sigma=1.0)
AGGRESSIVE_SPEC = ShiftSpec(degrees=75.0, blur_kernel=7, blur_sigma=2.0)

PRESETS = {
    "mild": MILD_SPEC,
    "aggressive": AGGRESSIVE_SPEC,
}


def get_preset_spec(name: str) -> ShiftSpec:
    if name not in PRESETS:
        raise KeyError(
            f"Unknown shift preset '{name}'. Available: {list(PRESETS.keys())}"
        )
    return PRESETS[name]


def get_preset(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Legacy helper — returns a deterministic shift *function* (not a Compose of
    stochastic layers) matching a named preset.
    """
    return build_shift(get_preset_spec(name))


def custom_rotation_blur(degrees: float, blur_kernel: int, blur_sigma: float = 1.0) -> Callable[[torch.Tensor], torch.Tensor]:
    if blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be odd.")
    return build_shift(ShiftSpec(
        degrees=degrees, blur_kernel=blur_kernel, blur_sigma=blur_sigma,
    ))
