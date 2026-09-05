"""
Auditable sample container + result records.

Every Cascade sample carries its dataset index so that drift measurements can
be traced back to the exact clean/shifted image pair and prediction that
produced them. This avoids the class of bugs where the Nth entry of a
misclassified list is silently paired with the Nth clean sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class Sample:
    """A single (clean, shifted) pair with identity + prediction metadata.

    Fields map directly to the four-group experiment design:
      - index:           original dataset index (clean == shifted → same image)
      - shifted_correct: True  → survived the shift (group: shifted-correct)
                         False → failed   the shift (group: shifted-incorrect)
    """

    index: int
    clean: torch.Tensor
    shifted: torch.Tensor
    true_label: int
    pred_label_shifted: int
    confidence_shifted: float = float("nan")
    entropy_shifted: float = float("nan")
    margin_shifted: float = float("nan")
    shifted_correct: bool = False
    clean_pred_label: Optional[int] = None
    clean_confidence: float = float("nan")
    extra: dict = field(default_factory=dict)

    def as_tuple(self):
        """Backwards-compatible 4-tuple for legacy callers."""
        return (self.clean, self.shifted, self.true_label, self.pred_label_shifted)


@dataclass
class SampleDrift:
    """Full drift record for one sample: index + D(k) per layer."""

    index: int
    layer_names: List[str]
    dk_true_raw: List[float]
    dk_true_norm: List[float]
    dk_pred_raw: List[float]
    dk_pred_norm: List[float]
    true_label: int
    pred_label: int
    group: str  # "shifted_correct" | "shifted_incorrect"
