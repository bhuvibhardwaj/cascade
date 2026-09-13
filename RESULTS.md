# PNR statistical layer — status against the math-backing document

Language matches "Mathematical Backing for the Population-Calibrated
Point-of-No-Return (PNR) Thresholds": **STATISTICALLY JUSTIFIED** means the
method is implemented and, once run, yields a valid finite-sample guarantee
or hypothesis-test conclusion; **NOT ESTABLISHED** means the claim is not
supported by these procedures; **EMPIRICALLY OBSERVED** means a measured
pattern, not a proof.

The paired `ttest_rel` / Wilcoxon tests in `cascade/report.py` compare D(k−1)
vs D(k) *within* a trajectory. They are not the Mann–Whitney test below and
were left unchanged.

## Code (synthetic verification)

| Claim | Status |
| --- | --- |
| Each θ_k approximates the true population quantile within DKW ε (single layer) or ε_sim (all L as a system). | **STATISTICALLY JUSTIFIED** — `dkw_epsilon`, `dkw_epsilon_simultaneous`; Monte Carlo coverage check in `cascade/tests/test_bounds.py`. |
| Shifted-vs-null D(k) populations differ at a layer (MWU + BH-FDR), as a method. | **STATISTICALLY JUSTIFIED** — `mann_whitney_layer_test`, `benjamini_hochberg`, `benjamini_yekutieli`, `layer_significance_table`. MWU matches scipy on synthetic data; BH reproduces the textbook 15-p-value example (reject first four at q=0.05). |
| Any single input's crossing D(k) > θ_k is individually significant. | **NOT ESTABLISHED BY MWU** — needs an individual-level tail probability or conformal guarantee (Section 4.1). |
| Naive per-layer 95th percentiles give a 5% false-PNR rate for PNR(x)=min{k:D(k)>θ_k}. | **NOT ESTABLISHED** — under independent synthetic layers, L=8, measured union FAR is ~33.7% as in Section 3.2. |
| Option B (Bonferroni, 1−α/L quantile) and Option C (empirical joint calibration) control trajectory FAR on synthetic data. | **STATISTICALLY JUSTIFIED** as methods; **EMPIRICALLY OBSERVED** on synthetic data: B ≈ 5% under independence; C ≈ 5% and tighter than B under correlated layers. |
| Amplification / early-PNR curve; compounding vs depth-correlated sensitivity. | **EMPIRICALLY OBSERVED** (growth) / **CAUSALLY UNVALIDATED** (Section 7 not implemented). |

`calibrate_pnr_thresholds()` is still the **naive** path. `diagnose()` now prints `calibration_method`, ε, and ε_sim so naive thresholds cannot be presented as Bonferroni/joint.

Option C searches a single multiplier on a naive (1−α)-quantile *template*. `calibrate_thresholds_joint` takes that pattern as `pairs` (template) plus disjoint `tune_pairs` (search for m). Omitting `tune_pairs` overfits and warns. For numbers that get quoted, holdout FAR is reported with a Wilson interval.

## Real data (MNIST, 8-layer CNN, aggressive shift)

Run: `python experiments/exp02_pnr_statistical_layer.py`

Artifacts: `results/pnr_statistical_layer.json`, `results/layer_significance.csv`, and `results/pnr_statistical_layer_mnist_seed42_1789105697/`.

Setup: EightLayerCNN trained 2 epochs on MNIST (seed 42, MPS). L=8 conv layers. Null = two Gaussian views of the same clean image (σ=0.05). Shifted = clean vs `aggressive` `ShiftSpec`. Three disjoint null folds plus a shifted MWU sample:

- calib n = 500 (θ_k)
- tune n = 350 (Option C multiplier; template frozen from calib)
- holdout n = 350 (union FAR + Wilson 95% CI — the number to cite)
- shifted n = 350 (MWU)

The previous n=80 run is not cited. At n_holdout=8, 1/8 = 12.5% has Wilson interval [2.2%, 47.1%]; that is why a bare percentage was not a result.

### DKW (actual n_pairs = 500, matching the PDF worked example)

n = 500, L = 8, δ = 0.05:

- ε = 0.0607
- ε_sim = 0.0759

A 95% target has true null CDF coverage within about ±6.1 percentile points per layer, or ±7.6 points if all eight thresholds are presented jointly. **STATISTICALLY JUSTIFIED**, and now numerically the same as the document's illustration.

### Trajectory false-alarm rate (Section 3.2) — holdout, Wilson 95% CI

Independence formula 1−0.95⁸ = 0.3366. Layers are not independent, so the formula is a reference, not a prediction.

| Method | Holdout |
| --- | --- |
| Naive 95th | 91/350 = 26.0% [95% CI: 21.7%–30.8%] |
| Bonferroni 99.375th (Option B) | 20/350 = 5.7% [95% CI: 3.7%–8.7%] |
| Joint (Option C, m ≈ 2.02 on a disjoint tune fold) | 14/350 = 4.0% [95% CI: 2.4%–6.6%] |

**EMPIRICALLY OBSERVED:** naive union FAR is high (CI well above 5%; below the independent-layers 33.7% figure, as the doc expected under correlation). Option B and Option C both have holdout CIs that cover 5%. That is compatible with a calibrated trajectory FAR at this n; it is not a proof that the deployed detector sits at exactly 5%. The interval width (~5–6 points) is what n=350 buy you around a 5% rate.

### Layer significance (MWU + BH / BY)

One-sided MWU of shifted D(k) vs holdout-null D(k) at each layer, then BH and BY at q=0.05. All eight layers rejected under both corrections (BH-adjusted p from ~4.6×10⁻¹⁰⁴ to ~1.1×10⁻¹¹⁵).

**STATISTICALLY JUSTIFIED** as a population test, and **verified-on-real-data** for this MNIST + aggressive-shift setup. This still **does not** make any one image's PNR crossing individually significant.

The ~16× layer-1→8 growth curve lives on the ResNet18/CIFAR-10 setup below, not on this MNIST toy CNN.

## Real data (CIFAR-10, ResNet18, runner-mild shift)

Run: `python experiments/exp03_pnr_statistical_layer_resnet18.py`

Artifacts: `results/pnr_statistical_layer_resnet18.json`, `results/layer_significance_resnet18.csv`, and `results/pnr_statistical_layer_resnet18_seed42_1789106507/`.

This is not a rerun of exp02 on different pixels. Same CIFAR ResNet18 as `train_and_Run_resnet18.py` (`conv1` 3×3 s1 p1, `maxpool=Identity`, 10-way `fc`, `max_layers=8`). Shift is the runner mild spec (45°, blur 5 / σ=1.5) then CIFAR normalize. Null is two-view Gaussian noise in pixel [0,1] (σ=0.05), then the same normalize — not identical-image pairs.

Folds were sized **before** launch (session budget: 1500 GradCAM pairs). All five are disjoint:

- calib n = 300 (θ_k only)
- tune n = 300 (Option C multiplier; naive template frozen from calib — **refit**, not MNIST m≈2.02)
- holdout n = 300 (union FAR + Wilson 95% CI — the number to cite)
- mwu_null n = 300 / mwu_shifted n = 300 (MWU+BH only; not used to pick θ_k, m, or FAR)

Layers (`Cascade` 8-layer sample): `conv1`, `layer1.1.conv1`, `layer2.0.conv1`, `layer2.1.conv1`, `layer3.0.conv2`, `layer3.1.conv2`, `layer4.0.conv2`, `layer4.1.conv2`. Trained 5 epochs, seed 42, MPS.

### DKW (n_pairs = 300)

n = 300, L = 8, δ = 0.05:

- ε = 0.0784
- ε_sim = 0.0981

Wider than the n=500 PDF illustration, as expected from the budget choice. **STATISTICALLY JUSTIFIED** at this n.

### Trajectory false-alarm rate (Section 3.2) — holdout, Wilson 95% CI

Independence formula 1−0.95⁸ = 0.3366 again. Option C multiplier on the **ResNet18 tune fold**: m ≈ 1.42 (tune FAR 15/300 = 5.0%; search notes empty — it hit the target on tune). Do not carry the MNIST m≈2.02.

| Method | Holdout |
| --- | --- |
| Naive 95th | 84/300 = 28.0% [95% CI: 23.2%–33.3%] |
| Bonferroni 99.375th (Option B) | 15/300 = 5.0% [95% CI: 3.1%–8.1%] |
| Joint (Option C, m ≈ 1.42 on a disjoint tune fold) | 19/300 = 6.3% [95% CI: 4.1%–9.7%] |

**EMPIRICALLY OBSERVED:** naive union FAR is again high (CI well above 5%; still below the independent-layers 33.7% figure). It is **not** closer to 5% than MNIST was, so this holdout does not by itself show a stronger “layers fire together” union effect than the 8-conv net. The joint **multiplier** did change (1.42 vs 2.02); that is why it was refit.

Option B’s holdout point estimate is 5.0%; Option C’s is 6.3%. Both CIs cover 5%. That is compatible with calibrated trajectory FAR at n=300; it is not a proof the detector sits at exactly 5%, and the 5% target was not backfilled. Interval width is ~5 points around a 5% rate.

### Layer significance (MWU + BH / BY)

One-sided MWU of mwu_shifted D(k) vs mwu_null D(k) at each layer (arrays already collected — no second GradCAM pass), then BH and BY at q=0.05. All eight layers rejected under both (BH-adjusted p from ~4.2×10⁻⁵⁰ to ~1.1×10⁻⁹⁶).

**STATISTICALLY JUSTIFIED** as a population test on this CIFAR-10 + mild-shift setup. Still **does not** make any one image’s PNR crossing individually significant.

Descriptive mean D_norm on the MWU folds is in the JSON (null decreasing with depth; shifted remaining O(1) under this mild shift). That is **not** a causal compounding test; Section 7 was not run.
