"""Evaluate original model A and the strong-form checkpoint on common CPU grids.

Both checkpoints are loaded without creating an optimizer or updating weights.
The default weak checkpoint is the original model A, not the separately
retrained seed-42 model used in the normalization repeat experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def boundary_diagnostics(model, problem, dtype, n=401):
    import numpy as np
    from mixed_weak_pcrb.validation import predict_fields

    margin = float(problem.output_scale_margin)
    peaks = np.asarray(problem.output_scales["moment"], dtype=float) / margin
    if margin <= 0 or np.any(~np.isfinite(peaks[:2])) or np.any(peaks[:2] <= 0):
        raise ValueError("Boundary moment normalization requires positive prior peaks")
    s = np.linspace(0.0, 1.0, n)
    edges = (
        (np.c_[np.zeros(n), s * problem.ly], 0),
        (np.c_[np.full(n, problem.lx), s * problem.ly], 0),
        (np.c_[s * problem.lx, np.zeros(n)], 1),
        (np.c_[s * problem.lx, np.full(n, problem.ly)], 1),
    )
    displacements, moments, normalized = [], [], []
    for points, component in edges:
        fields = predict_fields(model, points, dtype)
        values = np.asarray(fields["moment"][:, component], dtype=float)
        moments.append(values)
        normalized.append(values / peaks[component])
        displacements.append(np.asarray(fields["w"], dtype=float))
    moments, normalized = np.concatenate(moments), np.concatenate(normalized)
    return {
        "max_abs_w": float(np.max(np.abs(np.concatenate(displacements)))),
        "max_abs_Mnn": float(np.max(np.abs(moments))),
        "rms_Mnn": float(np.sqrt(np.mean(moments ** 2))),
        "normalized_max_abs_Mnn": float(np.max(np.abs(normalized))),
        "normalized_rms_Mnn": float(np.sqrt(np.mean(normalized ** 2))),
        "moment_prior_peak_scales": {
            "x_normal_edges_Mxx": float(peaks[0]), "y_normal_edges_Myy": float(peaks[1]),
        },
        "samples_per_edge": n,
        "rms_definition": "pooled RMS of four uniform endpoint-inclusive edge sample arrays",
        "normalization_source": "problem.output_scales['moment']/problem.output_scale_margin",
    }


def evaluate_checkpoint(config, weights, output, model_id, *, grid_size=201,
                        strong_grid_size=81, test_order=8, quadrature_order=14):
    import json
    import numpy as np
    import tensorflow as tf
    from mixed_weak_pcrb.model import build_model
    from mixed_weak_pcrb.problem import build_problem, resolve_output_scales, resolve_residual_scales
    from mixed_weak_pcrb.quadrature import RectMesh
    from mixed_weak_pcrb.strong_form import PHYSICS_KEYS, prior_residual_metrics, write_json
    from mixed_weak_pcrb import validation

    weights, output = Path(weights), Path(output)
    before = hashlib.sha256(weights.read_bytes()).hexdigest()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    tf.keras.backend.clear_session()
    tf.keras.backend.set_floatx(config["dtype"])
    dc = config["domain"]
    problem = build_problem(config)
    scales, _ = resolve_output_scales(config, problem)
    residual_scales, scale_source = resolve_residual_scales(config, problem)
    if residual_scales is None:
        raise ValueError("Common-prior evaluation requires enabled fixed prior scales")
    metadata_path = weights.parent / "problem_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        saved = metadata.get("network_output_scales")
        if saved is not None:
            for key, value in scales.items():
                if key not in saved or not np.allclose(value, saved[key], rtol=1e-12, atol=1e-15):
                    raise ValueError(f"Checkpoint/config output-scale mismatch for {key}")
    model = build_model(**config["network"], lx=dc["lx"], ly=dc["ly"], output_scales=scales)
    td = tf.as_dtype(config["dtype"])
    model(tf.zeros((1, 2), td), training=False)
    model.load_weights(str(weights))
    mesh = RectMesh(dc["lx"], dc["ly"], dc["nx"], dc["ny"])
    x, y = np.linspace(0, dc["lx"], grid_size), np.linspace(0, dc["ly"], grid_size)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xy = np.c_[xx.ravel(), yy.ravel()]
    predicted = validation.predict_fields(model, xy, config["dtype"])
    # The reference is first accessed here, after loading the fixed checkpoint.
    exact = problem.exact_numpy(xy)
    field_metrics = validation.field_error_metrics(predicted, exact)
    peaks = validation.peak_response_metrics(xy, predicted, exact, problem.lx, problem.ly)
    center = np.array([[dc["lx"] / 2, dc["ly"] / 2]])
    center_pred = validation.predict_fields(model, center, config["dtype"])["w"][0]
    center_ref = problem.exact_numpy(center)["w"][0]
    sx = np.linspace(0, dc["lx"], strong_grid_size + 2)[1:-1]
    sy = np.linspace(0, dc["ly"], strong_grid_size + 2)[1:-1]
    sxx, syy = np.meshgrid(sx, sy, indexing="xy")
    interior = np.c_[sxx.ravel(), syy.ravel()]
    prior_metrics, raw = prior_residual_metrics(
        model, problem, tf.constant(interior, td), residual_scales,
    )
    response_metrics = validation.strong_residual_metrics(
        model, problem, interior, config["dtype"],
    )
    weak_metrics, cell_residual, _ = validation.weak_validation(
        model, problem, mesh, config["dtype"], test_order, quadrature_order,
    )
    is_strong = model_id == "strong_gpu_seed42"
    method = "first_order_eight_field_strong" if is_strong else "mixed_local_weak"
    report = {
        "model_id": model_id, "method": method,
        "field_errors": field_metrics, "peak_response": peaks,
        "center_deflection": {"predicted": float(center_pred), "exact": float(center_ref),
            "relative_error": float(abs(center_pred - center_ref) / (abs(center_ref) + 1e-30))},
        "strong_residuals": response_metrics,
        "boundary_conditions": boundary_diagnostics(model, problem, config["dtype"]),
        "global_balance": validation.global_balance(model, problem, config["dtype"], order=64),
        "independent_weak_residuals": weak_metrics,
        "independent_point_residuals_prior": prior_metrics,
        "point_residual_normalization": {
            "source": "same fixed a-priori per-component scales as training; independent interior grid",
            "definition": "RMS over grid points and components within each equation group; ratio, not percent",
            "prior_source": scale_source,
            "scales": {k: np.asarray(v).tolist() for k, v in residual_scales.items()},
        },
        "quadrature_configuration": {
            "mode": "fixed", "cell_count": len(mesh.cells),
            "training_order": int(config["weak_form"]["quadrature_order"]),
            "training_test_modes_per_direction": None if is_strong else int(config["weak_form"]["test_order"]),
            "training_loss_form": method,
            "validation_order": quadrature_order, "validation_test_modes_per_direction": test_order,
        },
        "coordinate_mapping": problem.coordinate_mapping,
        "network_output_scales": {k: np.asarray(v).tolist() for k, v in scales.items()},
        "validation_settings": {"field_grid": grid_size, "strong_grid": strong_grid_size,
            "test_order": test_order, "quadrature_order": quadrature_order,
            "boundary_points_per_edge": 401, "global_balance_order": 64},
        "evaluation_provenance": {
            "checkpoint_sha256": before, "parameters": int(model.count_params()),
            "device": "CPU", "network_precision": config["dtype"],
            "field_error_accumulation": "float64", "optimizer_updates": 0,
            "weak_baseline": "original model A from the local-load case and ablation table",
        },
    }
    if hashlib.sha256(weights.read_bytes()).hexdigest() != before:
        raise RuntimeError("Checkpoint changed during evaluation")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", report)
    validation.save_metric_csv(output / "field_metrics.csv", field_metrics)
    validation.save_metric_csv(output / "peak_response_metrics.csv", peaks)
    validation.save_field_csv(output / "field_values.csv", xy, predicted, exact)
    validation.save_cell_diagnostics(output / "cell_weak_residuals.csv", mesh, cell_residual)
    columns = ["xi", "eta", "kinematic_x", "kinematic_y", "constitutive_xx",
               "constitutive_yy", "constitutive_xy", "moment_x", "moment_y", "equilibrium"]
    np.savetxt(output / "point_residuals.csv", np.column_stack([interior, *[raw[k] for k in PHYSICS_KEYS]]),
               delimiter=",", header=",".join(columns), comments="")
    if hasattr(problem, "dimensionalize_fields"):
        validation.save_field_csv(
            output / "physical_field_values.csv", problem.computational_to_physical(xy),
            problem.dimensionalize_fields(predicted), problem.dimensionalize_fields(exact),
        )
    return report


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-config", type=Path, default=root / "configs/local_load_plate.json")
    parser.add_argument("--strong-config", type=Path, default=root / "configs/local_strong.json")
    parser.add_argument("--weak-weights", type=Path, default=root / "pretrained/local_load_plate/model.weights.h5")
    parser.add_argument("--strong-weights", type=Path, default=root / "pretrained/local_strong/model.weights.h5")
    parser.add_argument("--output", type=Path, default=Path("validation/strong_control"))
    parser.add_argument("--grid-size", type=int, default=201)
    parser.add_argument("--strong-grid-size", type=int, default=81)
    parser.add_argument("--test-order", type=int, default=8)
    parser.add_argument("--quadrature-order", type=int, default=14)
    args = parser.parse_args()
    if min(args.grid_size, args.strong_grid_size, args.test_order, args.quadrature_order) < 2:
        parser.error("All grid sizes and orders must be at least two")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory is not empty: {args.output}")
    for weights in (args.weak_weights, args.strong_weights):
        if not weights.is_file():
            parser.error(f"Checkpoint not found: {weights}")
    os.environ.update({"TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1",
                       "OMP_NUM_THREADS": "2", "TF_ENABLE_ONEDNN_OPTS": "1"})
    from mixed_weak_pcrb.config import load_config
    from mixed_weak_pcrb.runtime import configure_device
    environment = configure_device("cpu")
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    environment.update({"intra_op_threads": 2, "inter_op_threads": 1, "optimizer_updates": 0,
                        "xla_evaluation": False, "onednn": True})
    from mixed_weak_pcrb.strong_form import write_json
    weak_config, strong_config = load_config(args.weak_config), load_config(args.strong_config)
    for key in ("seed", "dtype", "domain", "physics", "network", "output_scale", "weak_form"):
        if weak_config.get(key) != strong_config.get(key):
            raise ValueError(f"Control configurations differ in {key}; common evaluation is not defined")
    if weak_config.get("physics", {}).get("problem") != "local_gaussian":
        raise ValueError("This control evaluation is for the local Gaussian-load problem")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run_state.json").open("x", encoding="utf-8") as stream:
        stream.write('{"status":"evaluating"}')
    write_json(args.output / "environment.json", environment)
    reports = {}
    try:
        for model_id, config, weights in (
            ("original_A", weak_config, args.weak_weights),
            ("strong_gpu_seed42", strong_config, args.strong_weights),
        ):
            with tf.device("/CPU:0"):
                reports[model_id] = evaluate_checkpoint(
                    config, weights, args.output / model_id, model_id,
                    grid_size=args.grid_size, strong_grid_size=args.strong_grid_size,
                    test_order=args.test_order, quadrature_order=args.quadrature_order,
                )
            print(f"Evaluated {model_id}; checkpoint unchanged", flush=True)
        write_json(args.output / "comparison.json", {
            "models": {key: value["evaluation_provenance"] for key, value in reports.items()},
            "validation_settings": reports["original_A"]["validation_settings"],
            "normalization": reports["original_A"]["point_residual_normalization"],
            "statistical_scope": "one seed per formulation; no mean or standard deviation",
            "timing_scope": "evaluation does not measure training time; historical device conditions differ",
            "loss_note": "training losses and differently normalized weak residuals are not compared directly",
        })
        write_json(args.output / "run_state.json", {"status": "complete", "optimizer_updates": 0})
    except BaseException as exc:
        write_json(args.output / "run_state.json", {"status": "failed", "error": repr(exc)})
        raise
    print(f"Common control evaluation written to: {args.output}")


if __name__ == "__main__":
    main()
