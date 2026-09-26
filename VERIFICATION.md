# Verification record

Date: 2026-09-26.

## Implementation checks

- Eight unit tests pass: analytical polynomial residuals on a rectangle; component and area reductions; a parameter gradient checked against finite differences; common-prior RMS; archived file hashes; original-A/retrained-seed identity; canonical metric records; and rejection of a zero-epoch training request.
- All 14 manifest model entries (13 distinct checkpoints and the original-A alias) load successfully and produce finite fields. Four-branch A/B have 24,056 parameters; eight-branch C/D have 42,872.
- Historical and current B/C/D implementations produce identical eight-field predictions at the inspected points with the same archived weights.
- One-step CPU checks complete for B (normalization disabled), C (eight scalar branches) and the first-order strong baseline. The strong baseline's initial weight hash and first-step loss match the recorded implementation.

## Numerical reproduction

The three canonical checkpoints were reevaluated on CPU using the full 201 × 201 field grid, 81 × 81 point-residual grid, expanded test space and G=14 validation quadrature. Across 36 primary field, peak and residual quantities per case, the maximum absolute discrepancy from the archived values was below 1.2 × 10⁻¹⁰. This is smaller than the precision displayed in the chapter.

The original-A and strong-form checkpoints were also reevaluated together. Field errors, peak errors, common-prior pointwise residuals, weak residuals, boundary conditions and force balance reproduce the retained control records. The response-normalized constitutive diagnostic for original A differs by approximately 1.4 × 10⁻¹⁰ due to derivative reduction order; the comparison uses the unchanged common-prior diagnostic.

Table regeneration verifies 16 input-record hashes and experiment identities. Independent checks against the retained numerical records passed for 278 values and checkpoint hashes. All 12 CSV snapshots match newly generated files. Figure 4.14 was inspected for labels, units and data correspondence.

## Scope

Canonical and supplementary checkpoints are retained without optimizer updates. The one-step runs are installation/implementation checks in separate output directories and do not enter the published statistics. Repeated training across hardware and library versions can differ; execution metadata and final-epoch checkpoint hashes identify each reported result.
