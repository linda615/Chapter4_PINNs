"""Reproduce Chapter 4 tables and Figure 4.14 from the saved numerical records.

This script does not import TensorFlow, evaluate checkpoints, or train models.
Errors stored as ratios are converted to percent once. Normalization statistics
use three independently trained CPU runs and sample standard deviation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIELDS = ("w", "beta_x", "beta_y", "M_xx", "M_yy", "M_xy", "Q_x", "Q_y")
GROUPS = ("kinematic", "constitutive", "moment", "equilibrium")
SEEDS = (42, 7, 2026)


def load(relative):
    return json.loads((RESULTS / relative).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(output, name, rows):
    with (output / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def field_rows(record):
    rows = {row["field"]: row for row in record["field_errors"]}
    require(set(rows) == set(FIELDS), "A record does not contain exactly eight physical fields")
    for field in FIELDS:
        require(math.isfinite(rows[field]["relative_l2"]) and rows[field]["relative_l2"] >= 0,
                f"Invalid relative L2 error for {field}")
    return rows


def errors_percent(record):
    return {field: 100.0 * row["relative_l2"] for field, row in field_rows(record).items()}


def peak_rows(record):
    rows = {row["field"]: row for row in record["peak_response"]}
    require(set(rows) == set(FIELDS), "A record does not contain eight peak-response rows")
    return rows


def verify_inputs():
    provenance = load("provenance.json")
    for source in provenance["files"]:
        path = RESULTS / source["file"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == source["sha256"], f"Result record hash mismatch: {source['file']}")
    metadata = load("run_metadata.json")
    require(hashlib.sha256((RESULTS / "run_metadata.json").read_bytes()).hexdigest() == provenance["run_metadata_sha256"],
            "Run metadata hash mismatch")
    local = load("raw/canonical/local.json")
    a = load("raw/ablation/A.json")
    require(local == a, "Table 4.13 A must use the canonical local-load metrics")
    original_a = load("raw/strong_control/original_A.json")
    for field, value in errors_percent(a).items():
        require(abs(value - errors_percent(original_a)[field]) < 1e-5,
                "The strong-control weak baseline must match original A within evaluation precision")
    require(metadata["original_A"]["weights_sha256"] == metadata["ablation_A"]["weights_sha256"],
            "Original A checkpoint identity mismatch")
    require(metadata["seed42_normalized"]["weights_sha256"] != metadata["ablation_A"]["weights_sha256"],
            "The independent CPU seed42 must not be replaced by original A")
    strong = load("raw/strong_control/strong_gpu_seed42.json")
    require(strong["point_residual_normalization"] == original_a["point_residual_normalization"],
            "Strong and weak point residuals must use identical prior scales")
    require(strong["network_output_scales"] == original_a["network_output_scales"],
            "Strong and weak output scales differ")
    for model, record in [("original_A", original_a), ("strong_gpu_seed42", strong)]:
        require(record["validation_settings"]["field_grid"] == 201 and record["validation_settings"]["strong_grid"] == 81,
                f"Unexpected evaluation grids for {model}")
        for group in GROUPS:
            for metric in ("normalized_rms", "normalized_max_absolute"):
                value = record["independent_point_residuals_prior"][group][metric]
                require(math.isfinite(value) and value >= 0, f"Invalid {group} {metric}")
    for seed in SEEDS:
        for variant in ("normalized", "unnormalized"):
            run_id = f"seed{seed}_{variant}"
            record = load(f"raw/normalization/{run_id}.json")
            run = metadata[run_id]
            require(record["seed"] == seed and record["variant"] == variant, f"Incorrect identity for {run_id}")
            require(record["epochs"] == 20000 and run["training_cost"]["epochs"] == 20000,
                    f"Incomplete normalization training: {run_id}")
            require(run["training_device"] == "CPU" and record["grid"] == "201x201",
                    f"Unexpected device or grid for {run_id}")
            require(record["training_seconds_recorded"] == run["training_cost"]["elapsed_seconds"],
                    f"Inconsistent training time for {run_id}")
            require(set(record["relative_l2_percent"]) == set(FIELDS), f"Missing field for {run_id}")
            require(all(math.isfinite(x) and x >= 0 for x in record["relative_l2_percent"].values()),
                    f"Invalid field error for {run_id}")
    require(metadata["strong_gpu_seed42"]["training_cost"]["epochs"] == 20000,
            "Strong training must be complete")
    return metadata, len(provenance["files"])


def canonical_tables(output):
    chapter3 = load("raw/chapter3_comparison.json")["relative_l2_percent"]
    for case, table in [("sinusoidal", "4_3"), ("local", "4_7"), ("heterogeneous", "4_10")]:
        record = load(f"raw/canonical/{case}.json")
        errors, peaks = errors_percent(record), peak_rows(record)
        rows = []
        for field in FIELDS:
            row = {"field": field, "relative_l2_percent": errors[field],
                   "relative_peak_amplitude_error_percent": 100 * peaks[field]["relative_peak_error"],
                   "normalized_peak_location_error_percent": 100 * peaks[field]["normalized_location_error"]}
            if case == "sinusoidal":
                row["chapter3_relative_l2_percent"] = chapter3.get(field, "")
            rows.append(row)
        write_csv(output, f"table_{table}.csv", rows)


def ablation_tables(output, metadata):
    records = {model: load(f"raw/ablation/{model}.json") for model in "ABCD"}
    errors = {model: errors_percent(record) for model, record in records.items()}
    write_csv(output, "table_4_13.csv", [{"model": model, **errors[model]} for model in "ABCD"])
    effects = []
    for field in FIELDS:
        a, b, c, d = (errors[model][field] for model in "ABCD")
        effects.append({"field": field,
                        "gamma_RMS": math.sqrt((b / a) * (d / c)),
                        "gamma_Branch": math.sqrt((c / a) * (d / b))})
    write_csv(output, "table_4_13_geometric_effects.csv", effects)
    rows = []
    for model, record in records.items():
        run = metadata[f"ablation_{model}"]
        rows.append({"model": model, "parameters": run["parameters"],
                     "training_seconds": run["training_cost"]["elapsed_seconds"],
                     "center_deflection_relative_error_percent": 100 * record["center_deflection"]["relative_error"],
                     "global_balance_relative_error_percent": 100 * record["global_balance"]["relative_error"]})
    write_csv(output, "table_4_14.csv", rows)


def normalization_table(output):
    records = {(seed, variant): load(f"raw/normalization/seed{seed}_{variant}.json")
               for seed in SEEDS for variant in ("normalized", "unnormalized")}
    raw_rows, summary = [], []
    for (seed, variant), record in records.items():
        for field in FIELDS:
            raw_rows.append({"seed": seed, "variant": variant, "field": field,
                             "relative_l2_percent": record["relative_l2_percent"][field]})
    for field in FIELDS:
        values = {variant: [records[seed, variant]["relative_l2_percent"][field] for seed in SEEDS]
                  for variant in ("normalized", "unnormalized")}
        means = {variant: statistics.mean(data) for variant, data in values.items()}
        summary.append({"field": field,
                        "normalized_mean_percent": means["normalized"],
                        "normalized_sample_sd_percent": statistics.stdev(values["normalized"]),
                        "unnormalized_mean_percent": means["unnormalized"],
                        "unnormalized_sample_sd_percent": statistics.stdev(values["unnormalized"]),
                        "ratio_of_means_unnormalized_to_normalized": means["unnormalized"] / means["normalized"],
                        "seeds_where_normalized_better": sum(a < b for a, b in zip(values["normalized"], values["unnormalized"])),
                        "n": len(SEEDS), "sd_ddof": 1})
    write_csv(output, "normalization_individual_runs.csv", raw_rows)
    write_csv(output, "table_4_15.csv", summary)


def control_tables(output):
    records = {method: load(f"raw/strong_control/{method}.json")
               for method in ("strong_gpu_seed42", "original_A")}
    errors, residuals, balance = [], [], []
    for method, record in records.items():
        peaks = peak_rows(record)
        for field, value in errors_percent(record).items():
            errors.append({"model": method, "field": field, "relative_l2_percent": value,
                           "relative_peak_amplitude_error_percent": 100 * peaks[field]["relative_peak_error"],
                           "normalized_peak_location_error_percent": 100 * peaks[field]["normalized_location_error"]})
        for group in GROUPS:
            values = record["independent_point_residuals_prior"][group]
            residuals.append({"model": method, "equation_group": group,
                              "normalized_rms": values["normalized_rms"],
                              "normalized_max_absolute": values["normalized_max_absolute"],
                              "unit": "dimensionless ratio"})
        balance.append({"model": method, **record["global_balance"],
                        "relative_error_percent": 100 * record["global_balance"]["relative_error"]})
    write_csv(output, "table_4_16.csv", errors)
    write_csv(output, "figure_4_14_values.csv", residuals)
    write_csv(output, "strong_control_global_balance.csv", balance)
    return records


def runtime_table(output, metadata):
    rows = []
    for run_id, record in metadata.items():
        rows.append({"run_id": run_id, "seed": record["seed"], "parameters": record["parameters"],
                     "epochs": record["training_cost"]["epochs"],
                     "training_seconds": record["training_cost"]["elapsed_seconds"],
                     "training_device": record["training_device"], "evaluation_device": record["evaluation_device"],
                     "checkpoint_alias_of": record.get("checkpoint_alias_of", ""),
                     "note": "Recorded runtime; execution environments differ. Not a controlled speed comparison."})
    write_csv(output, "recorded_training_costs.csv", rows)


def residual_figure(output, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.2, 3.7), layout="constrained")
    width = 0.32
    styles = [("strong_gpu_seed42", -0.5, "#CB6B28", "First-order strong form"),
              ("original_A", 0.5, "#2468A0", "Local weak form (model A)")]
    for method, offset, color, label in styles:
        values = [records[method]["independent_point_residuals_prior"][group]["normalized_rms"] for group in GROUPS]
        bars = ax.bar([i + offset * width for i in range(4)], values, width, color=color, label=label, zorder=3)
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], fontsize=8, padding=3)
    ax.set_xticks(range(4), ["Kinematic", "Constitutive", "Moment-shear", "Transverse equilibrium"])
    ax.set_ylabel("RMS / common prior scale (dimensionless)")
    ax.set_ylim(0, 0.061)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(output / "figure_4_14.png", dpi=300)
    fig.savefig(output / "figure_4_14.svg", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "derived", help="Output directory (default: repository/derived)")
    parser.add_argument("--no-figure", action="store_true", help="Generate CSV files using only the Python standard library")
    args = parser.parse_args()
    output = args.output.resolve()
    require(output != RESULTS.resolve() and RESULTS.resolve() not in output.parents,
            "Choose an output directory outside results/ to preserve the source records")
    metadata, verified_count = verify_inputs()
    output.mkdir(parents=True, exist_ok=True)
    canonical_tables(output)
    ablation_tables(output, metadata)
    normalization_table(output)
    records = control_tables(output)
    runtime_table(output, metadata)
    if not args.no_figure:
        residual_figure(output, records)
    report = {"result_records_verified": verified_count, "original_A_identity_verified": True,
              "independent_CPU_seed42_identity_verified": True, "common_prior_residual_scales_verified": True,
              "normalization_seeds": list(SEEDS), "sample_standard_deviation_ddof": 1,
              "field_grid": "201x201", "point_residual_grid": "81x81",
              "operation": "statistics and plots from saved numerical metrics; no model evaluation or training"}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {verified_count} source records. Tables and figures written to {output}")


if __name__ == "__main__":
    main()
