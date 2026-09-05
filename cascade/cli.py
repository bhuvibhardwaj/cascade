"""
Command-line interface. Produces a reproducible analysis run, not just a
stdout profile: for each invocation it writes config.json, metrics.json, and
layer-wise drift CSVs under an output directory. Output uses deterministic
functional shifts (no stochastic RandomRotation layers).

Usage:
    cascade analyze \
        --model path/to/model.pt \
        --dataset cifar10 \
        --data-root ./data \
        --shift aggressive \
        --n-samples 300 \
        --out-dir results/run1 \
        --device cuda
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from .core import Cascade
from .plots import save_layer_profiles, save_summary_json
from .report import fragility_profile
from .samples import Sample
from .shift import get_preset_spec, make_shifted_dataset

DATASETS = {
    "cifar10": torchvision.datasets.CIFAR10,
    "mnist": torchvision.datasets.MNIST,
}


def _load_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _load_model(args) -> torch.nn.Module:
    obj = torch.load(args.model, map_location=args.device)
    if isinstance(obj, torch.nn.Module):
        return obj
    if not args.model_class:
        raise ValueError(
            "Checkpoint appears to be a state_dict. Pass --model-class "
            "(e.g. torchvision.models.resnet18) to reconstruct the architecture."
        )
    model_cls = _load_class(args.model_class)
    model = model_cls(num_classes=args.num_classes) if args.num_classes else model_cls()
    model.load_state_dict(obj)
    return model


def _predict_all(model, ds_clean, ds_shift, device):
    """Evaluate the shifted test set, returning Samples indexed by dataset idx."""
    model.eval()
    samples: list[Sample] = []
    with torch.no_grad():
        for i in range(len(ds_clean)):
            clean_img, true_lbl = ds_clean[i]
            shifted_img, _ = ds_shift[i]
            s_out = model(shifted_img.unsqueeze(0).to(device))
            s_probs = torch.softmax(s_out, dim=1)
            s_conf, s_pred = torch.max(s_probs, 1)
            top2 = s_probs.topk(min(2, s_probs.shape[1])).values.squeeze(0)
            margin = float((top2[0] - top2[1]).item()) if len(top2) >= 2 else 0.0
            entropy = float(-torch.sum(s_probs * torch.log(s_probs + 1e-9)).item())
            samples.append(
                Sample(
                    index=i,
                    clean=clean_img.unsqueeze(0),
                    shifted=shifted_img.unsqueeze(0),
                    true_label=int(true_lbl),
                    pred_label_shifted=int(s_pred.item()),
                    confidence_shifted=float(s_conf.item()),
                    entropy_shifted=entropy,
                    margin_shifted=margin,
                    shifted_correct=(s_pred.item() == true_lbl),
                )
            )
    return samples


def cmd_analyze(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    device = args.device
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.dataset not in DATASETS:
        raise SystemExit(f"Unknown dataset '{args.dataset}'. Choices: {list(DATASETS.keys())}")

    dataset_cls = DATASETS[args.dataset]
    shift_spec = get_preset_spec(args.shift)

    to_tensor = T.ToTensor()
    ds_clean = dataset_cls(
        root=args.data_root, train=False, download=True, transform=to_tensor
    )
    ds_raw = dataset_cls(
        root=args.data_root, train=False, download=True, transform=None
    )
    ds_shift = make_shifted_dataset(ds_raw, shift_spec, to_tensor=True)

    model = _load_model(args).to(device).eval()

    print("Predicting shifted test set...")
    all_samples = _predict_all(model, ds_clean, ds_shift, device)
    shifted_correct = [s for s in all_samples if s.shifted_correct]
    shifted_incorrect = [s for s in all_samples if not s.shifted_correct]
    acc = len(shifted_correct) / max(1, len(all_samples))
    print(f"Shifted accuracy: {acc:.4f} ({len(shifted_correct)}/{len(all_samples)})")
    print(f"  correct: {len(shifted_correct)}   incorrect (failed): {len(shifted_incorrect)}")

    def subset(lst, n, seed):
        if len(lst) <= n:
            return list(lst)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(lst), size=n, replace=False)
        idx.sort()
        return [lst[int(i)] for i in idx]

    n = args.n_samples
    inc = subset(shifted_incorrect, n, seed)
    cor = subset(shifted_correct, n, seed)
    print(f"Profiling {len(inc)} shifted-incorrect + {len(cor)} shifted-correct samples.")

    cascade = Cascade(model, max_layers=args.max_layers, device=device)
    print(f"Layers: {cascade.layer_names}")
    profile = fragility_profile(
        cascade, samples_incorrect=inc, samples_correct=cor,
    )
    print()
    print(profile.summary())

    config = {
        "seed": seed,
        "device": device,
        "dataset": args.dataset,
        "shift": {"name": args.shift, **asdict(shift_spec)},
        "n_samples_per_group": n,
        "shifted_accuracy": acc,
        "n_shifted_correct": len(shifted_correct),
        "n_shifted_incorrect": len(shifted_incorrect),
        "layers": cascade.layer_names,
        "timestamp": int(time.time()),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    _ = save_summary_json(profile, os.path.join(out_dir, "metrics.json"))
    produced = save_layer_profiles(profile, out_dir)
    print(f"\nArtifacts saved to {out_dir}: {produced}")

    # Save predictions/metadata CSVs for the analyzed sample subset
    import csv as _csv
    meta = os.path.join(out_dir, "sample_metadata.csv")
    with open(meta, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["index", "group", "true_label", "pred_label", "confidence", "entropy", "margin"])
        for s in inc:
            w.writerow([s.index, "shifted_incorrect", s.true_label,
                        s.pred_label_shifted, s.confidence_shifted,
                        s.entropy_shifted, s.margin_shifted])
        for s in cor:
            w.writerow([s.index, "shifted_correct", s.true_label,
                        s.pred_label_shifted, s.confidence_shifted,
                        s.entropy_shifted, s.margin_shifted])
    print(f"Sample metadata: {meta}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cascade")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser(
        "analyze", help="Analyze a checkpoint under shift, save artifacts to --out-dir"
    )
    analyze.add_argument("--model", required=True)
    analyze.add_argument("--model-class", default=None)
    analyze.add_argument("--num-classes", type=int, default=None)
    analyze.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    analyze.add_argument("--data-root", default="./data")
    analyze.add_argument("--shift", default="aggressive", choices=["mild", "aggressive"])
    analyze.add_argument("--n-samples", type=int, default=300, help="samples per outcome group")
    analyze.add_argument("--max-layers", type=int, default=None)
    analyze.add_argument("--device", default="cpu")
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--out-dir", required=True)
    analyze.set_defaults(func=cmd_analyze)

    return parser


def main(argv=None):
    build_parser().parse_args(argv).func()


if __name__ == "__main__":
    main()
