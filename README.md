# PCRB-Net for plate bending

Reference implementation and numerical records for Chapter 4, *Local weak-form method for bending deformation of rock strata*. The model uses eight physical fields: deflection, two rotations, three bending moments and two shear forces. Four residual branches group these fields by physical role. The ablation configurations also support eight scalar branches.

## Method and cases

- First-order mixed equations with local Petrov–Galerkin residuals and analytical bubble test functions.
- Hard output transforms for the homogeneous simply supported boundary conditions used in the three cases.
- Component-wise output scales and optional prior-RMS residual normalization.
- Fixed tensor-product Gauss–Legendre training quadrature, `G=8`.
- Post-training verification on independent points, an expanded test space and higher-order quadrature.

| Section | Configuration | Output activation | Reference solution |
|---|---|---|---|
| 4.4.2: sinusoidal load | `configs/sinusoidal_plate.json` | Linear | Single-mode Navier solution |
| 4.4.3: local Gaussian load | `configs/local_load_plate.json` | tanh | 61 × 61 Navier series |
| 4.4.4: heterogeneous roof | `configs/heterogeneous_roof.json` | Linear | 33 × 33 sine Ritz approximation |

Hidden layers use tanh. Reference solutions are used after training for evaluation and do not supply interior training labels. Stored output scales are part of each experiment and must be retained when loading its checkpoint.

The archived checkpoints above retain their original configurations. The final
matched experiments used in the revised Section 4.5 are provided separately in
`configs/final/`: both formulations use the same eight-field architecture,
`G=8` points, residual schedules, pure problem-prior output scales with
`chi_f=1.5`, linear output heads and no data-dependent scale calibration.

## Installation

The experiments used Python 3.9 and TensorFlow 2.10.1. CPU records were obtained with NumPy 1.25.2; the local strong-form GPU run used NumPy 1.23.5. Networks use float32, while field-error norms accumulate in float64.

```sh
python -m pip install -r requirements.txt
```

Run commands from the repository root. Evaluation defaults to CPU with TF32 disabled. GPU training requires compatible TensorFlow/CUDA libraries; the recorded strong-form run used an NVIDIA GeForce RTX 3060 Laptop GPU, CUDA 11.2 and XLA.

## Reevaluate the three supplied models

```sh
python evaluate.py --config configs/sinusoidal_plate.json
python evaluate.py --config configs/local_load_plate.json
python evaluate.py --config configs/heterogeneous_roof.json
```

The field grid is `201 × 201`; pointwise residuals use `81 × 81` interior points. Weak-residual verification in the expanded test space uses eight test modes per direction and `G=14`. Boundary checks use 401 points per edge, and global force balance uses a 64-point Gauss rule.

Outputs in `validation/<case>/` include field and peak errors, strong and weak residuals, boundary and balance checks, field CSV files, figures and execution metadata. Reference metrics in `pretrained/<case>/` use the CPU convention of the final chapter. Checkpoint hashes and evaluation context accompany the records.

Choose a fresh directory with `--output` for another evaluation. For a short installation check:

```sh
python evaluate.py --config configs/sinusoidal_plate.json --grid-size 41 --strong-grid-size 21 --output validation/smoke_sine
```

Reduced grids must not replace the published metrics. Evaluator field plots use computational coordinates and fields; `physical_field_values.csv` provides dimensional quantities for mapped rectangular cases.

## Train the weak-form models

```sh
python train.py --config configs/sinusoidal_plate.json
python train.py --config configs/local_load_plate.json
python train.py --config configs/heterogeneous_roof.json
```

Each configuration specifies 20,000 epochs. The final epoch is saved without selection by reference-solution error. Training defaults to CPU; use `--device gpu` for a supported GPU or `--epochs 2 --output runs/smoke_sine` for a short check. Existing nonempty output directories are rejected.

Fresh training may differ across hardware and library builds. Reevaluate the supplied checkpoint to reproduce a particular recorded model; retraining assesses the specified training procedure.

## Section 4.5: ablation and repeated runs

| Model | Branches | Prior-RMS normalization | Configuration |
|---|---:|---|---|
| A | 4 | Enabled | `configs/ablation/A.json` |
| B | 4 | Disabled | `configs/ablation/B.json` |
| C | 8 | Enabled | `configs/ablation/C.json` |
| D | 8 | Disabled | `configs/ablation/D.json` |

```sh
python train_ablation.py --config configs/ablation/B.json
python train_ablation.py --config configs/ablation/C.json
python train_ablation.py --config configs/seeds/seed42_normalized.json
python train_ablation.py --config configs/seeds/seed42_unnormalized.json
```

The other repeat configurations use seeds `7` and `2026`. Six independent CPU runs underlie Table 4.15. Its seed-42 pair is a separate retraining batch; original A and B are excluded. Standard deviations use `ddof=1`, with `n=3` for each normalization setting.

Original A is `pretrained/local_load_plate/model.weights.h5`, shared by Section 4.4.3, Tables 4.13–4.14 and the final strong/weak comparison. B–D and all six CPU repeat checkpoints are included. `metadata/manifest.json` records configurations, relative paths, hashes and training provenance.

Historical B–D configurations retain inactive adaptive-quadrature fields. `train_ablation.py` accepts only fixed `G=8` with adaptation disabled, removes unused quadrature-penalty keys from runtime loss weights and records that conversion. The four active physical losses and schedules remain unchanged.

Evaluate an ablation by specifying its weights:

```sh
python evaluate.py --config configs/ablation/C.json --weights pretrained/ablation_C/model.weights.h5 --output validation/ablation_C
```

## First-order strong-form comparison

The strong baseline uses the same eight outputs, architecture, boundary transforms, prior scales and staged optimization as the local Gaussian weak-form case. Its four equation groups are enforced pointwise. This is a first-order comparison inspired by FO-PINN, rather than a reproduction of every network and benchmark in that publication.

The archived checkpoint comparison used the earlier explicit-scale, tanh-head configuration with seed 42. It remains available for exact reevaluation: GPU training took **420.6679645 s**, including first graph compilation, the training loop and logging, and excluding evaluation and checkpoint saving. Original A has a historical record of **252.0723079 s**, with incomplete device/thread/compiler metadata. These records do not support a controlled speed comparison.

`train_strong.py` trains a new model; `evaluate_control.py` reevaluates the archived strong checkpoint and original A on CPU:

```sh
python train_strong.py --device gpu --output runs/local_strong_new
python evaluate_control.py --output validation/strong_control_new
```

Use `--help` for device, checkpoint and output options. Evaluation reports eight field errors, peak errors, boundary and force balance, and independent pointwise residuals normalized by common prior scales.

For the final matched comparison, train both formulations with seeds 42, 7 and
2026. Each command writes to a distinct directory:

```sh
python train.py --device gpu --config configs/final/local_weak_seed42.json
python train.py --device gpu --config configs/final/local_weak_seed7.json
python train.py --device gpu --config configs/final/local_weak_seed2026.json
python train_strong.py --device gpu --config configs/final/local_strong_seed42.json
python train_strong.py --device gpu --config configs/final/local_strong_seed7.json
python train_strong.py --device gpu --config configs/final/local_strong_seed2026.json
```

The final Figure 4.14 record contains all six independently trained runs. Bars
show the mean and sample standard deviation (`ddof=1`, `n=3`), while the white
markers show individual random seeds. These runs use the final pure-prior,
linear-head configuration and must not be mixed with the archived seed-42
checkpoint comparison.

The comparison uses different pointwise normalization from the response-based normalization in the three case tables. Training losses and these two residual conventions must not be compared directly by magnitude. Conclusions apply to this case and its complete configurations.

## Regenerate tables and Figure 4.14

```sh
python reproduce_tables.py --output derived
```

This requires NumPy and Matplotlib, but no TensorFlow training. It uses numerical records in `results/`, recomputes the three-seed statistics and writes comparison tables and the independent-residual figure. Figure 4.14 uses the final matched strong/weak runs for seeds 42, 7 and 2026; the archived original-A single-seed comparison is emitted separately as `strong_control_single_seed_residuals.csv`.

The Chapter 3 column of Table 4.3 is retained as an archived same-problem comparison record. This repository reevaluates Chapter 4 checkpoints; the Chapter 3 training implementation is outside its scope.

## Verification

```sh
python -m unittest discover -s tests -v
```

Tests cover implementation properties and result consistency. Archived checkpoints, numerical records and timing metadata remain separate from fresh outputs in `runs/`, `validation/` and `derived/`.

See [VERIFICATION.md](VERIFICATION.md) for the checks performed on this version.

## Reference

Gladstone, R. J., Nabian, M. A., Sukumar, N., Srivastava, A., and Meidani, H. (2025). *FO-PINN: A First-Order formulation for Physics-Informed Neural Networks*. Engineering Analysis with Boundary Elements, 174, 106161. https://doi.org/10.1016/j.enganabound.2025.106161

## Terms

This repository is supplied for academic review and verification. The existing terms in [NOTICE.md](NOTICE.md) apply.
