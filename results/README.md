# Numerical results

This directory contains the saved numerical records used by the Chapter 4 tables. It includes the three benchmark cases, the A–D ablation, six CPU normalization runs, and the local Gaussian strong-form comparison.

## Regenerate tables and Figure 4.14

From the repository root:

```bash
python reproduce_tables.py
python reproduce_tables.py --output path/to/output
```

The default output directory is `derived/`. CSV generation uses the Python standard library; the figure requires Matplotlib. Use `--no-figure` to generate CSV files without Matplotlib. The script verifies the record hashes and experiment identities before calculating the tables. It does not train or evaluate a model.

`tables/` and `figures/` contain snapshots produced by this script. The CSV files retain full precision; rounding is applied only for manuscript display. Table 4.15 reports mean ± sample standard deviation with three decimal places. The raw metrics store dimensionless ratios unless a key explicitly ends in `_percent`. Figure 4.14 plots dimensionless residuals, not percentages.

## Data mapping

| Manuscript output | Source records |
| --- | --- |
| Table 4.3 | `raw/canonical/sinusoidal.json` and `raw/chapter3_comparison.json` |
| Tables 4.7 and 4.10 | `raw/canonical/local.json` and `raw/canonical/heterogeneous.json` |
| Tables 4.13–4.14 | `raw/ablation/A.json` through `D.json`; historical costs in `run_metadata.json` |
| Table 4.15 | All six `raw/normalization/seed*_*.json` records |
| Table 4.16 and Figure 4.14 | `raw/strong_control/strong_gpu_seed42.json` and `original_A.json` |

Model A is the original local-load checkpoint. Its metrics in `raw/ablation/A.json` are identical to `raw/canonical/local.json`. The weak baseline in Table 4.16 uses the same checkpoint, re-evaluated to add the common-prior point residuals without an optimizer update.

Table 4.15 uses a separate CPU training batch: seeds 42, 7 and 2026, each with and without normalization, 20,000 epochs per run. Its seed 42 results are new training runs, distinct from the original A and B checkpoints. Standard deviations are sample standard deviations (`ddof=1`, `n=3`); they are not confidence intervals. The A–D and strong-form comparisons each use single-run results.

The geometric effects accompanying Table 4.13 are computed separately for each field:

```text
Gamma_RMS    = sqrt((error_B / error_A) * (error_D / error_C))
Gamma_Branch = sqrt((error_C / error_A) * (error_D / error_B))
```

These are descriptive effects from the four saved runs, not estimates across random seeds.

## Evaluation and timing

Field errors use a 201 × 201 grid. The strong/weak comparison uses an independent 81 × 81 grid for point residuals, normalized by the same physical prior scales. Both comparison checkpoints were evaluated on CPU. The Chapter 3 comparison also uses CPU evaluation on the same 201 × 201 grid; unavailable rotation outputs are left blank, and the shear sign convention is aligned before comparison.

The new first-order strong-form run used an NVIDIA GeForce RTX 3060 Laptop GPU, float32, TF32 disabled and XLA. Its recorded training time is 420.6679645 s, including the first compiled training step and excluding validation and checkpoint saving. An unfinished CPU attempt was stopped for hardware migration; the GPU run restarted from the same seed 42 initialization for all 20,000 epochs. Only the completed final-epoch result is reported.

The original A–D training costs are retained from their historical runs. Their cost records do not identify the training device. The six normalization runs were trained on CPU. These timings and the new GPU timing are not a controlled speed comparison; the package does not calculate a speedup ratio. The original A runtime is 252.0723079 s, whereas the separately retrained normalized CPU seed 42 runtime is 3646.8103716 s.

## Provenance

`run_metadata.json` records checkpoint identities, SHA-256 hashes, training costs and device information. `provenance.json` maps the exported metrics to retained source records and preserves their hashes. Archive-relative identifiers in that file identify the original records; those archives are not dependencies for table regeneration. The six normalization records have their machine-specific `source_run` paths replaced by logical identifiers. All numerical values are unchanged.

Full prediction arrays are omitted from this compact results package. Table regeneration recomputes percentages, geometric effects, means and sample standard deviations from saved metrics; it does not independently reconstruct field errors or derivatives from arrays. Checkpoint evaluation is a separate operation using the repository evaluation commands.
