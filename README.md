# Cascade

A layer-wise attribution-drift diagnostic for CNNs under distribution shift.

Distribution shift doesn't cause failure all at once — attribution drift
`D(k)` usually grows with depth. Cascade measures this directly using
GradCAM attribution drift at every Conv2d layer of a model, and surfaces:

- **Fragility profiles** — per-layer `D(k)` distributions with confidence
  intervals, paired t-tests, and a non-parametric (Wilcoxon) fallback.
  Profiles are reported separately for the **shifted-correct** (survived)
  and **shifted-incorrect** (failed) groups so you can compare the two.
- **Per-input diagnosis** — for a single example, a calibrated *point of
  no return* (PNR) index, plus a stable/verdict on prediction-path
  coherence (see below on the interpretation for stable/unstable).

## Terminology note

This package measures **layer-wise attribution drift**, defined as

```
D_raw(k)  = || GradCAM_shifted(k) - GradCAM_clean(k) ||_2
D_norm(k) = D_raw(k) / ( || GradCAM_clean(k) ||_2 + eps )
```

`D(k)` is reported as `D_norm(k)` by default. This is *an attribution
distance* — we do not claim it is a direct measure of "spurious signal
strength" or "causal propagation." Those are stronger claims that require
experimental support (e.g. comparison of shifted-correct vs
shifted-incorrect trajectories, intervention experiments).

## Install

```bash
pip install -e .
```

## Python API

```python
from cascade import (
    Cascade,
    Sample,
    ShiftSpec,
    build_shift,
    calibrate_pnr_thresholds,
    diagnose,
    fragility_profile,
    make_shifted_dataset,
)

cascade = Cascade(model, device="cuda")   # auto-discovers Conv2d layers

# ---- Fragility profile (shifted-incorrect vs shifted-correct) ----
profile = fragility_profile(
    cascade,
    samples_incorrect=[Sample(...), ...],   # shifted samples that failed
    samples_correct  =[Sample(...), ...],   # shifted samples that survived
)
print(profile.summary())

# ---- Population-calibrated Point-of-No-Return (PNR) diagnosis ----
#
# PNR is only meaningful when the threshold is calibrated against a
# clean-population null. Without calibration, PNR is not computed (it
# is `None` for every sample), so the construct remains falsifiable.
null_pairs = [ (clean_i, clean_i, label_i) for i in ... ]   # same image vs itself
pnr_thresholds = calibrate_pnr_thresholds(cascade, null_pairs, quantile=0.95)

result = diagnose(
    cascade, clean_img, shifted_img, true_label=3, pred_label=7,
    threshold=pnr_thresholds,
)
print(result.summary())
```

## CLI

The CLI now writes reproducible artifacts, not just stdout.

```bash
cascade analyze \
    --model checkpoint.pt \
    --model-class torchvision.models.resnet18 \
    --num-classes 10 \
    --dataset cifar10 \
    --data-root ./data \
    --shift aggressive \
    --n-samples 300 \
    --seed 42 \
    --out-dir results/run1 \
    --device cuda
```

Outputs written to `results/run1/`:
```
config.json         seed, shift spec, dataset, layers, accuracy
metrics.json        per-group per-layer D_norm summaries + significance tests
sample_metadata.csv analyzed samples (index, group, label, conf, entropy, margin)
layer_drift.csv     raw + normalized D(k) per sample per layer
layer_dk_norm_mean.png  mean ± SD D_norm by layer (failed vs survived)
layer_dk_norm_box.png   median[IQR] D_norm by layer (failed vs survived)
```

## Experiment scripts

Two hardened runners are included:

- `train_and_run.py` — MNIST small CNN, 5 epochs
- `train_and_Run_resnet18.py` — ResNet18 on CIFAR-10, 5 epochs
- `experiments/exp02_pnr_statistical_layer.py` — DKW / MWU / BH / trajectory FAR on MNIST (cheap three-way-split check)
- `experiments/exp03_pnr_statistical_layer_resnet18.py` — same pipeline on ResNet18/CIFAR-10 (the setup to cite; default 300 per fold)

Both do:
1. **Dataset-indexed clean/shifted pairing** (`clean[i]` always matches `shifted[i]` — never by list position).
2. **Genuinely deterministic shifts** via `T.functional.rotate` + explicit `GaussianBlur` sigma (no `RandomRotation` stochastic layers used at access time).
3. **Seeds everywhere** (Python, NumPy, PyTorch, DataLoader generator) — recorded in `config.json`.
4. **Full reproducible artifacts**: checkpoint, config, environment, metrics, full predictions, plots, CSVs, run log.
5. **Four-group separation**: all shifted samples are scored, then `shifted_correct` vs `shifted_incorrect` are analyzed separately (not only failures).
6. **Calibrated PNR**: threshold derived from clean-vs-clean (same-image) null population at 95th percentile.

Run `python3 train_and_run.py` or `python3 train_and_Run_resnet18.py`. Results go to `results/<name>_<shift>_seed<s>_<timestamp>/`.

## Package layout

- `cascade/core.py` — GradCAM hooks + `D_raw(k)` and `D_norm(k)` computation
- `cascade/samples.py` — `Sample` (indexed, auditable) and `SampleDrift` dataclasses
- `cascade/shift.py` — deterministic `ShiftSpec`, `build_shift`, `make_shifted_dataset`
- `cascade/report.py` — fragility profiles (paired tests, two outcome groups)
- `cascade/diagnose.py` — per-input PNR + stable/unstable verdict
- `cascade/pnr.py` — population-calibrated PNR threshold calibration (naive quantile path)
- `cascade/bounds.py` — DKW ε/ε_sim, MWU, BH/BY, trajectory FAR, Bonferroni/joint PNR calibration
- `cascade/significance.py` — per-layer shifted-vs-null MWU + FDR table
- `cascade/plots.py` — layer-wise D(k) plots + CSVs
- `cascade/cli.py` — `cascade analyze` (artifact-producing, seeded, deterministic)

## Statistical layer (PNR)

See `RESULTS.md` for what the math-backing document claims vs what is now code-verified and measured on real data. Naive `calibrate_pnr_thresholds(..., quantile=0.95)` remains the default; it does **not** give a 5% false-PNR rate for the 8-layer union rule. Use `calibrate_thresholds_bonferroni` or `calibrate_thresholds_joint(..., tune_pairs=...)` (Option C overfits if you omit the disjoint tune fold), and `layer_significance_table` for the shifted-vs-null population test (distinct from the paired layer-to-layer tests in `report.py`).

```bash
python experiments/exp02_pnr_statistical_layer.py
python experiments/exp03_pnr_statistical_layer_resnet18.py
```
