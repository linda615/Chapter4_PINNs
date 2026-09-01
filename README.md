# PCRS-Net fixed-quadrature reference implementation

This repository is a compact review artifact for a mixed local weak-form physics-informed neural network for Kirchhoff plate bending. It contains only the fixed-quadrature implementation and the three configurations used to reproduce the reported verification cases.

## Included method components

- Four coupled output branches for deflection, rotation, bending moment, and shear force.
- First-order mixed-variable local weak residuals.
- Hard output transforms for homogeneous simply supported boundary conditions.
- Physics-based output scaling and prior RMS residual normalization.
- Fixed tensor-product Gauss–Legendre training quadrature with `G=8`.
- Independent post-training weak-residual verification with higher-order test modes and `G=14` quadrature.

The independent weak-residual check uses test-function and subdomain normalization. It is evaluated after training with fixed network parameters and does not apply the staged training weights.

## Verification cases

- `sinusoidal_plate`: simply supported plate under sinusoidal loading.
- `local_load_plate`: rectangular plate under a finite-width Gaussian load.
- `heterogeneous_roof`: nonhomogeneous roof plate with a localized stiffness-reduction zone.

Each JSON file under `configs/` is complete and self-contained. The small pretrained checkpoints and their reference metrics are under `pretrained/`.

## Environment

The checkpoints were verified with Python 3.9.13, TensorFlow 2.10.1, NumPy 1.25.2, and Matplotlib 3.7.1.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Evaluate the supplied checkpoints

Run these commands from the repository root:

```powershell
python evaluate.py --config configs/sinusoidal_plate.json
python evaluate.py --config configs/local_load_plate.json
python evaluate.py --config configs/heterogeneous_roof.json
```

Evaluation reports and figures are written to `validation/<case>/`. The default field grid is `201 × 201`; the independent weak-residual check uses eight test modes per coordinate direction and 14-point Gauss–Legendre quadrature in each direction.

For a faster installation check, reduce only the validation grids:

```powershell
python evaluate.py --config configs/sinusoidal_plate.json --grid-size 41 --strong-grid-size 21
```

## Train from scratch

```powershell
python train.py --config configs/sinusoidal_plate.json
python train.py --config configs/local_load_plate.json
python train.py --config configs/heterogeneous_roof.json
```

Training artifacts are written to `runs/<case>/`. A short smoke run can be requested without editing a configuration:

```powershell
python train.py --config configs/sinusoidal_plate.json --epochs 2
```

Because neural-network optimization can vary across hardware and software builds, small numerical differences from the supplied reference metrics are expected. The random seed, network architecture, physical parameters, loss schedules, and fixed quadrature settings are recorded in each configuration.

## Repository scope

This is a restricted academic-review snapshot, not an open-source distribution. See [NOTICE.md](NOTICE.md) for the applicable terms.
