"""
Stage 1 — shift-severity sweep.

Objective (per the research roadmap): find a shift regime on the existing
ResNet18/CIFAR-10 checkpoint where the failure rate among clean-correct
samples lands in roughly [20%, 80%], with n_survived >= 50, so that Stage 2
(failed-vs-survived D(k) comparison) and Stage 4 (activation-patching pilot)
have enough survivors to compare against.

This does NOT retrain anything. It loads the checkpoint already saved by
exp03 (results/pnr_statistical_layer_resnet18_seed42_*/resnet18_cifar10.pt),
sweeps deterministic ShiftSpec parameters, and reports clean accuracy,
shifted accuracy, and — the number that actually matters here — the failure
rate AMONG CLEAN-CORRECT samples (not overall accuracy), since that's the
population Exp02/03/04's calibration and MVE folds are drawn from.

Run with:
    python stage1_severity_sweep.py --checkpoint <path/to/resnet18_cifar10.pt>

No thresholds are calibrated or fit here — this is purely a measurement
pass. Per the "do not threshold-hunt" rule in the roadmap, the shift(s)
selected here should be frozen before Stage 2/4 begin, not re-picked after
seeing later results.
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

# Make the cascade package importable regardless of cwd (mirrors exp03's
# pattern for finding the repo root; adjust if your layout differs).
_REPO_ROOT = Path(__file__).resolve().parent
for candidate in (_REPO_ROOT, _REPO_ROOT / "cascade", _REPO_ROOT.parent / "cascade"):
    if (candidate / "cascade" / "__init__.py").exists():
        sys.path.insert(0, str(candidate))
        break

from cascade.shift import ShiftSpec, build_shift  # noqa: E402


def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """Same architecture surgery as exp03: 3x3 stem, no maxpool, 512->10 fc."""
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


@dataclass(frozen=True)
class SweepPoint:
    degrees: float
    blur_kernel: int
    blur_sigma: float
    noise_std: float = 0.0


# Pre-registered sweep grid. Decide this BEFORE looking at any results and
# do not add points after seeing the table — that's threshold-hunting on
# the shift itself. Edit this list once, before running, not iteratively.
SWEEP_GRID: List[SweepPoint] = [
    SweepPoint(degrees=0, blur_kernel=0, blur_sigma=0.0),      # clean sanity check
    SweepPoint(degrees=5, blur_kernel=3, blur_sigma=0.5),
    SweepPoint(degrees=10, blur_kernel=3, blur_sigma=0.5),
    SweepPoint(degrees=15, blur_kernel=3, blur_sigma=1.0),
    SweepPoint(degrees=20, blur_kernel=5, blur_sigma=1.0),
    SweepPoint(degrees=25, blur_kernel=5, blur_sigma=1.0),
    SweepPoint(degrees=30, blur_kernel=5, blur_sigma=1.5),
    SweepPoint(degrees=35, blur_kernel=5, blur_sigma=1.5),
    SweepPoint(degrees=45, blur_kernel=5, blur_sigma=1.5),      # Exp03's "runner_mild" — known FR=93.8%
]


def evaluate_shift(
    model: nn.Module,
    clean_ds,
    point: SweepPoint,
    device: torch.device,
    normalize: T.Normalize,
) -> Tuple[int, int, int, int]:
    """Return (n_total, n_clean_correct, n_survived, n_failed)."""
    shift_fn = build_shift(
        ShiftSpec(
            degrees=point.degrees,
            blur_kernel=point.blur_kernel,
            blur_sigma=point.blur_sigma,
            noise_std=point.noise_std,
        )
    )

    n_total = 0
    n_clean_correct = 0
    n_survived = 0
    n_failed = 0

    model.eval()
    with torch.no_grad():
        for img, label in clean_ds:
            n_total += 1
            img = img.to(device)
            label_t = torch.tensor([label], device=device)

            clean_input = normalize(img).unsqueeze(0)
            clean_logits = model(clean_input)
            clean_pred = clean_logits.argmax(dim=1)
            if clean_pred.item() != label:
                continue
            n_clean_correct += 1

            shifted_img = shift_fn(img)
            shifted_input = normalize(shifted_img).unsqueeze(0)
            shifted_logits = model(shifted_input)
            shifted_pred = shifted_logits.argmax(dim=1)

            if shifted_pred.item() == label:
                n_survived += 1
            else:
                n_failed += 1

    return n_total, n_clean_correct, n_survived, n_failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to resnet18_cifar10.pt")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Cap on clean test images evaluated per sweep point (full 10k test set works too, just slower).",
    )
    parser.add_argument("--out", default="results/stage1_severity_sweep.json")
    parser.add_argument("--out-csv", default="results/stage1_severity_sweep.csv")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device={device}")

    model = build_resnet18_cifar().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state if not isinstance(state, dict) or "state_dict" not in state
                           else state["state_dict"])
    model.eval()

    normalize = T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    to_tensor = T.ToTensor()

    full_test = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=to_tensor
    )
    if args.limit is not None and args.limit < len(full_test):
        indices = list(range(args.limit))
        clean_ds = torch.utils.data.Subset(full_test, indices)
    else:
        clean_ds = full_test

    rows = []
    for point in SWEEP_GRID:
        n_total, n_clean_correct, n_survived, n_failed = evaluate_shift(
            model, clean_ds, point, device, normalize
        )
        failure_rate = n_failed / n_clean_correct if n_clean_correct else float("nan")
        clean_acc = n_clean_correct / n_total if n_total else float("nan")

        row = {
            **asdict(point),
            "n_total": n_total,
            "n_clean_correct": n_clean_correct,
            "clean_accuracy": clean_acc,
            "n_survived": n_survived,
            "n_failed": n_failed,
            "failure_rate": failure_rate,
            "in_target_band": bool(0.20 <= failure_rate <= 0.80 and n_survived >= 50)
            if n_clean_correct else False,
        }
        rows.append(row)
        print(
            f"deg={point.degrees:5.1f} blur_k={point.blur_kernel} sigma={point.blur_sigma:4.1f} "
            f"| clean_acc={clean_acc:.3f} | FR={failure_rate:.3f} | n_survived={n_survived} "
            f"| {'<-- CANDIDATE' if row['in_target_band'] else ''}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"sweep": rows}, f, indent=2)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    candidates = [r for r in rows if r["in_target_band"]]
    print("\n--- candidates (failure_rate in [0.20, 0.80] and n_survived >= 50) ---")
    if not candidates:
        print("NONE. Widen the sweep grid (finer steps near the transition) or reconsider the model/shift family.")
    for r in candidates:
        print(
            f"  deg={r['degrees']} blur_k={r['blur_kernel']} sigma={r['blur_sigma']} "
            f"-> FR={r['failure_rate']:.3f}, n_survived={r['n_survived']}"
        )

    print(f"\nWrote {args.out} and {args.out_csv}")


if __name__ == "__main__":
    main()
