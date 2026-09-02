from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from .model import build_model, require_tf
from .problem import build_problem, resolve_output_scales
from .quadrature import RectMesh
from .weak_form import batched_cell_residuals, batch_jacobian


FIELD_NAMES = ("w", "beta_x", "beta_y", "M_xx", "M_yy", "M_xy", "Q_x", "Q_y")
FIELD_PLOT_LABELS = (r"$w$", r"$\beta_x$", r"$\beta_y$", r"$M_{xx}$", r"$M_{yy}$", r"$M_{xy}$", r"$Q_x$", r"$Q_y$")
REPRESENTATIVE_FIELD_INDICES = (0, 5, 6)  # w, M_xy, Q_x


def _dtype(name):
    tf = require_tf()
    return tf.float64 if name == "float64" else tf.float32


def flatten_fields(fields):
    return np.column_stack([
        fields["w"], fields["beta"][:, 0], fields["beta"][:, 1],
        fields["moment"][:, 0], fields["moment"][:, 1], fields["moment"][:, 2],
        fields["shear"][:, 0], fields["shear"][:, 1],
    ])


def predict_fields(model, xy, dtype="float64", batch_size=8192):
    tf = require_tf()
    pieces = {key: [] for key in ("w", "beta", "moment", "shear")}
    for start in range(0, len(xy), batch_size):
        out = model(tf.convert_to_tensor(xy[start:start+batch_size], _dtype(dtype)), training=False)
        for key in pieces:
            value = out[key].numpy()
            pieces[key].append(value[:, 0] if key == "w" else value)
    return {key: np.concatenate(value, axis=0) for key, value in pieces.items()}


def field_error_metrics(predicted, exact):
    pred, ref = flatten_fields(predicted), flatten_fields(exact)
    rows = []
    for i, name in enumerate(FIELD_NAMES):
        error = pred[:, i]-ref[:, i]
        ref_l2 = np.linalg.norm(ref[:, i])
        ref_peak = np.max(np.abs(ref[:, i]))
        rows.append({
            "field": name,
            "relative_l2": float(np.linalg.norm(error)/(ref_l2+1e-30)),
            "max_absolute": float(np.max(np.abs(error))),
            "normalized_linf": float(np.max(np.abs(error))/(ref_peak+1e-30)),
            "rmse": float(np.sqrt(np.mean(error**2))),
        })
    return rows


def strong_residual_metrics(model, problem, xy, dtype="float64", batch_size=2048):
    """Independent pointwise check of all four mixed equations."""
    tf, td = require_tf(), _dtype(dtype)
    sums = {k: 0.0 for k in ("kinematic", "constitutive", "moment", "equilibrium")}
    maxima = dict(sums)
    scales = dict(sums)
    count = 0
    for start in range(0, len(xy), batch_size):
        points = tf.convert_to_tensor(xy[start:start+batch_size], td)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(points)
            out = model(points, training=False)
            w, beta, moment, shear = out["w"], out["beta"], out["moment"], out["shear"]
        gw = batch_jacobian(tape, w, points, "validation/w")[:, 0, :]
        gb = batch_jacobian(tape, beta, points, "validation/beta")
        gm = batch_jacobian(tape, moment, points, "validation/moment")
        gq = batch_jacobian(tape, shear, points, "validation/shear")
        del tape
        target_m = problem.constitutive_target_tf(gb, points)
        metric_x, metric_y = problem.divergence_factors
        residuals = {
            "kinematic": beta-gw,
            "constitutive": moment-target_m,
            "moment": tf.stack([
                shear[:, 0]+metric_x*gm[:, 0, 0]+metric_y*gm[:, 2, 1],
                shear[:, 1]+metric_x*gm[:, 2, 0]+metric_y*gm[:, 1, 1],
            ], axis=1),
            "equilibrium": metric_x*gq[:, 0, 0]+metric_y*gq[:, 1, 1]-problem.load_tf(points),
        }
        reference = {
            "kinematic": beta,
            "constitutive": moment,
            "moment": shear,
            "equilibrium": problem.load_tf(points),
        }
        for key, value in residuals.items():
            arr = value.numpy()
            scale = reference[key].numpy()
            sums[key] += float(np.sum(arr**2))
            maxima[key] = max(maxima[key], float(np.max(np.abs(arr))))
            scales[key] += float(np.sum(scale**2))
        count += len(points)
    return {key: {"rms": np.sqrt(sums[key]/count), "max_absolute": maxima[key], "normalized_rms": np.sqrt(sums[key]/(scales[key]+1e-30))} for key in sums}


def boundary_metrics(model, problem, dtype="float64", n=401):
    s = np.linspace(0.0, 1.0, n)
    lx, ly = problem.lx, problem.ly
    edges = {
        "x0": np.c_[np.zeros(n), s*ly], "xL": np.c_[np.full(n, lx), s*ly],
        "y0": np.c_[s*lx, np.zeros(n)], "yL": np.c_[s*lx, np.full(n, ly)],
    }
    out = {name: predict_fields(model, xy, dtype) for name, xy in edges.items()}
    return {
        "max_abs_w": float(max(np.max(np.abs(v["w"])) for v in out.values())),
        "max_abs_Mnn": float(max(np.max(np.abs(out["x0"]["moment"][:, 0])), np.max(np.abs(out["xL"]["moment"][:, 0])), np.max(np.abs(out["y0"]["moment"][:, 1])), np.max(np.abs(out["yL"]["moment"][:, 1])))),
    }


def global_balance(model, problem, dtype="float64", order=64):
    from numpy.polynomial.legendre import leggauss
    z, weights = leggauss(order)
    lx, ly = problem.lx, problem.ly
    x, wx = (z+1)*lx/2, weights*lx/2
    y, wy = (z+1)*ly/2, weights*ly/2
    left = predict_fields(model, np.c_[np.zeros(order), y], dtype)["shear"][:, 0]
    right = predict_fields(model, np.c_[np.full(order, lx), y], dtype)["shear"][:, 0]
    bottom = predict_fields(model, np.c_[x, np.zeros(order)], dtype)["shear"][:, 1]
    top = predict_fields(model, np.c_[x, np.full(order, ly)], dtype)["shear"][:, 1]
    metric_x, metric_y = problem.divergence_factors
    boundary_flux = metric_x*np.sum(wy*(-left+right))+metric_y*np.sum(wx*(-bottom+top))
    xx, yy = np.meshgrid(x, y, indexing="ij")
    load = problem.load_numpy(np.c_[xx.ravel(), yy.ravel()]).reshape(order, order)
    total_load = np.sum(wx[:, None]*wy[None, :]*load)
    return {"boundary_flux": float(boundary_flux), "total_load": float(total_load), "relative_error": float(abs(boundary_flux-total_load)/(abs(total_load)+1e-30))}


def weak_validation(model, problem, mesh, dtype="float64", test_order=4, quadrature_order=12):
    residuals = batched_cell_residuals(model, problem, mesh.cells, quadrature_order, test_order, dtype, training=False)
    arrays = {key: value.numpy() for key, value in residuals.items()}
    per_cell = np.zeros(len(mesh.cells))
    report = {}
    for key, value in arrays.items():
        axes = tuple(range(1, value.ndim))
        local = np.mean(value**2, axis=axes)
        per_cell += local
        report[key] = {
            "rms": float(np.sqrt(np.mean(value**2))),
            "max_absolute": float(np.max(np.abs(value))),
        }
    return report, np.sqrt(per_cell), arrays


def save_field_csv(path, xy, predicted, exact):
    pred, ref = flatten_fields(predicted), flatten_fields(exact)
    header = ["x", "y"]+[f"pred_{n}" for n in FIELD_NAMES]+[f"exact_{n}" for n in FIELD_NAMES]+[f"error_{n}" for n in FIELD_NAMES]
    np.savetxt(path, np.c_[xy, pred, ref, pred-ref], delimiter=",", header=",".join(header), comments="")


def save_metric_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def peak_response_metrics(xy, predicted, reference, lx=1.0, ly=1.0):
    """Compare peak magnitudes and locations for all eight predicted fields."""
    pred, ref = flatten_fields(predicted), flatten_fields(reference)
    diagonal = np.hypot(lx, ly)
    rows = []
    for index in range(len(FIELD_NAMES)):
        pred_index = int(np.argmax(np.abs(pred[:, index])))
        pred_peak = float(np.abs(pred[pred_index, index]))
        ref_absolute = np.abs(ref[:, index])
        ref_peak = float(np.max(ref_absolute))
        # Symmetric bending fields can have several physically equivalent
        # absolute peaks. Measure against the nearest reference peak instead of
        # whichever one np.argmax happens to return first.
        candidates = np.flatnonzero(ref_absolute >= (1.0-1.0e-6)*ref_peak)
        distances = np.linalg.norm(xy[candidates]-xy[pred_index], axis=1)
        ref_index = int(candidates[int(np.argmin(distances))])
        rows.append({
            "field": FIELD_NAMES[index],
            "predicted_peak_abs": pred_peak,
            "reference_peak_abs": ref_peak,
            "relative_peak_error": abs(pred_peak-ref_peak)/(ref_peak+1.0e-30),
            "predicted_peak_x": float(xy[pred_index, 0]),
            "predicted_peak_y": float(xy[pred_index, 1]),
            "reference_peak_x": float(xy[ref_index, 0]),
            "reference_peak_y": float(xy[ref_index, 1]),
            "normalized_location_error": float(np.min(distances)/(diagonal+1.0e-30)),
        })
    return rows


def save_cell_diagnostics(path, mesh, residuals):
    rows = []
    for cell, residual in zip(mesh.cells, residuals):
        rows.append({
            "x_min": float(cell[0]), "x_max": float(cell[1]),
            "y_min": float(cell[2]), "y_max": float(cell[3]),
            "weak_residual": float(residual),
        })
    save_metric_csv(path, rows)


def plot_representative_fields(output, grid_shape, extent, predicted_values, reference_values,
                               reference_label="解析解", coordinate_labels=(r"$x$", r"$y$")):
    """Plot w, M_xy and Q_x as analytical, predicted and absolute-error fields."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    output = Path(output)
    ny, nx = grid_shape
    row_labels = (reference_label, "PCRB-Net预测值", "绝对误差")
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.5), constrained_layout=True)
    for col, idx in enumerate(REPRESENTATIVE_FIELD_INDICES):
        exact_data = reference_values[:, idx]
        predicted_data = predicted_values[:, idx]
        error_data = np.abs(predicted_data-exact_data)
        field_min = min(float(np.min(exact_data)), float(np.min(predicted_data)))
        field_max = max(float(np.max(exact_data)), float(np.max(predicted_data)))
        for row, data in enumerate((exact_data, predicted_data, error_data)):
            limits = {"vmin": field_min, "vmax": field_max} if row < 2 else {}
            image = axes[row, col].imshow(
                data.reshape(ny, nx), origin="lower", extent=extent,
                aspect="equal", cmap="viridis" if row < 2 else "magma", **limits,
            )
            axes[row, col].set_title(f"{row_labels[row]} {FIELD_PLOT_LABELS[idx]}")
            if row == 2:
                axes[row, col].set_xlabel(coordinate_labels[0])
            if col == 0:
                axes[row, col].set_ylabel(coordinate_labels[1])
            fig.colorbar(image, ax=axes[row, col], shrink=.82)
    figure_path = output/"representative_fields_w_Mxy_Qx.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    return figure_path


def plot_results(
    output, problem, model, dtype, grid_shape, xy, predicted, exact,
    cell_residual, mesh,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    output = Path(output)
    pred, ref = flatten_fields(predicted), flatten_fields(exact)
    ny, nx = grid_shape
    extent = [0, problem.lx, 0, problem.ly]
    coordinate_labels = problem.coordinate_labels
    plot_representative_fields(
        output, grid_shape, extent, pred, ref,
        reference_label=problem.reference_label,
        coordinate_labels=coordinate_labels,
    )

    xline = np.linspace(0, problem.lx, 501)
    line_xy = np.c_[xline, np.full_like(xline, problem.ly/2)]
    line_pred = flatten_fields(predict_fields(model=model, xy=line_xy, dtype=dtype))
    line_ref = flatten_fields(problem.exact_numpy(line_xy))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, idx in zip(axes.ravel(), (0, 1, 3, 6)):
        ax.plot(xline, line_ref[:, idx], "k-", label=problem.reference_label)
        ax.plot(xline, line_pred[:, idx], "r--", label="PCRB-Net预测值")
        transverse = r"$\eta=0.5$" if coordinate_labels[1] == r"$\eta$" else r"$y=L_y/2$"
        ax.set_title(f"{FIELD_PLOT_LABELS[idx]} 沿 {transverse} 的分布"); ax.set_xlabel(coordinate_labels[0]); ax.grid(alpha=.3); ax.legend()
    fig.savefig(output/"centerline_comparison.png", dpi=180); plt.close(fig)

    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    def cell_plot(ax, values, cmap):
        patches = [Rectangle((x0, y0), x1-x0, y1-y0) for x0, x1, y0, y1 in mesh.cells]
        collection = PatchCollection(patches, cmap=cmap, edgecolor="white", linewidth=.25)
        collection.set_array(np.asarray(values, dtype=float))
        ax.add_collection(collection); ax.set_xlim(0, problem.lx); ax.set_ylim(0, problem.ly)
        ax.set_aspect("equal")
        return collection

    fig, residual_axis = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    im = cell_plot(residual_axis, cell_residual, "magma")
    residual_axis.set_xlabel(coordinate_labels[0]); residual_axis.set_ylabel(coordinate_labels[1])
    fig.colorbar(im, ax=residual_axis)
    fig.savefig(output/"weak_residual_distribution.png", dpi=300); plt.close(fig)

    fig, load_axis = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    load_image = load_axis.imshow(
        exact["load"].reshape(ny, nx), origin="lower", extent=extent,
        aspect="equal", cmap="magma",
    )
    load_axis.set_title(getattr(problem, "load_plot_title", "局部荷载分布"))
    load_axis.set_xlabel(coordinate_labels[0]); load_axis.set_ylabel(coordinate_labels[1])
    fig.colorbar(load_image, ax=load_axis)
    fig.savefig(output/"load_distribution.png", dpi=180); plt.close(fig)

    if hasattr(problem, "rigidity_numpy"):
        rigidity = problem.rigidity_numpy(xy).reshape(ny, nx)
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4), constrained_layout=True)
        rigidity_image = axes[0].imshow(
            rigidity, origin="lower", extent=extent, aspect="equal", cmap="cividis",
        )
        axes[0].set_title(r"无量纲弯曲刚度 $\bar D(\xi,\eta)$")
        load_image = axes[1].imshow(
            exact["load"].reshape(ny, nx), origin="lower", extent=extent,
            aspect="equal", cmap="magma",
        )
        axes[1].set_title(getattr(problem, "load_plot_title", "荷载分布"))
        for ax in axes:
            ax.set_xlabel(coordinate_labels[0]); ax.set_ylabel(coordinate_labels[1])
        fig.colorbar(rigidity_image, ax=axes[0])
        fig.colorbar(load_image, ax=axes[1])
        fig.savefig(output/"material_and_load_distribution.png", dpi=240)
        plt.close(fig)


def evaluate(
    config,
    grid_size=201,
    strong_grid_size=81,
    validation_test_order=8,
    validation_quadrature_order=14,
    weights_path=None,
    validation_output=None,
):
    tf = require_tf()
    tf.keras.backend.set_floatx(config.get("dtype", "float64"))
    dc, nc = config["domain"], config["network"]
    problem = build_problem(config)
    mesh = RectMesh(dc["lx"], dc["ly"], dc["nx"], dc["ny"])
    output = Path(config["output_dir"])
    weights_path = Path(weights_path) if weights_path else output/"model.weights.h5"
    metadata_path = weights_path.parent/"problem_metadata.json"
    output_scales, _ = resolve_output_scales(config, problem)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        saved_scales = metadata.get("network_output_scales")
        if saved_scales is not None:
            output_scales = {
                name: np.asarray(value, dtype=float)
                for name, value in saved_scales.items()
            }
    model = build_model(**nc, lx=dc["lx"], ly=dc["ly"], output_scales=output_scales)
    model(tf.zeros((1, 2), dtype=_dtype(config.get("dtype", "float64"))), training=False)
    if not weights_path.exists():
        raise FileNotFoundError(f"Trained weights not found: {weights_path}")
    model.load_weights(weights_path)
    validation_output = (
        Path(validation_output) if validation_output else output/"validation"
    )
    validation_output.mkdir(parents=True, exist_ok=True)

    x, y = np.linspace(0, dc["lx"], grid_size), np.linspace(0, dc["ly"], grid_size)
    xx, yy = np.meshgrid(x, y, indexing="xy"); xy = np.c_[xx.ravel(), yy.ravel()]
    predicted, exact = predict_fields(model, xy, config["dtype"]), problem.exact_numpy(xy)
    field_metrics = field_error_metrics(predicted, exact)
    peak_metrics = peak_response_metrics(xy, predicted, exact, problem.lx, problem.ly)
    center = np.array([[dc["lx"]/2, dc["ly"]/2]])
    center_pred, center_ref = predict_fields(model, center, config["dtype"])["w"][0], problem.exact_numpy(center)["w"][0]

    sx, sy = np.linspace(0, dc["lx"], strong_grid_size+2)[1:-1], np.linspace(0, dc["ly"], strong_grid_size+2)[1:-1]
    sxx, syy = np.meshgrid(sx, sy, indexing="xy")
    strong = strong_residual_metrics(model, problem, np.c_[sxx.ravel(), syy.ravel()], config["dtype"])
    boundary = boundary_metrics(model, problem, config["dtype"])
    balance = global_balance(model, problem, config["dtype"])
    weak, cell_residual, _ = weak_validation(model, problem, mesh, config["dtype"], validation_test_order, validation_quadrature_order)

    report = {
        "field_errors": field_metrics,
        "peak_response": peak_metrics,
        "center_deflection": {"predicted": float(center_pred), "exact": float(center_ref), "relative_error": float(abs(center_pred-center_ref)/(abs(center_ref)+1e-30))},
        "strong_residuals": strong, "boundary_conditions": boundary, "global_balance": balance,
        "independent_weak_residuals": weak,
        "quadrature_configuration": {
            "mode": "fixed",
            "cell_count": len(mesh.cells),
            "training_order": int(config["weak_form"]["quadrature_order"]),
            "training_test_modes_per_direction": int(config["weak_form"]["test_order"]),
            "validation_order": int(validation_quadrature_order),
            "validation_test_modes_per_direction": int(validation_test_order),
        },
        "coordinate_mapping": problem.coordinate_mapping,
        "network_output_scales": {
            name: np.asarray(value).tolist() for name, value in output_scales.items()
        },
        "validation_settings": {"field_grid": grid_size, "strong_grid": strong_grid_size, "test_order": validation_test_order, "quadrature_order": validation_quadrature_order},
    }
    if hasattr(problem, "rigidity_numpy"):
        rigidity = problem.rigidity_numpy(xy)
        report["material_heterogeneity"] = {
            "normalized_rigidity_min": float(np.min(rigidity)),
            "normalized_rigidity_max": float(np.max(rigidity)),
            "normalized_rigidity_mean": float(np.mean(rigidity)),
        }
        np.savetxt(
            validation_output/"material_field_values.csv",
            np.c_[xy, rigidity, exact["load"]], delimiter=",",
            header="xi,eta,normalized_rigidity,normalized_load", comments="",
        )
    if hasattr(problem, "ritz_data"):
        report["reference_solution"] = {
            "method": "sine Ritz",
            "modes_per_direction": int(problem.reference_modes),
            "quadrature_order": int(problem.reference_quadrature_order),
            "relative_algebraic_residual": float(problem.ritz_data["relative_algebraic_residual"]),
        }
    (validation_output/"metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    save_metric_csv(validation_output/"field_metrics.csv", field_metrics)
    save_metric_csv(validation_output/"peak_response_metrics.csv", peak_metrics)
    save_field_csv(validation_output/"field_values.csv", xy, predicted, exact)
    save_cell_diagnostics(validation_output/"cell_weak_residuals.csv", mesh, cell_residual)
    if hasattr(problem, "dimensionalize_fields"):
        physical_xy = problem.computational_to_physical(xy)
        save_field_csv(
            validation_output/"physical_field_values.csv", physical_xy,
            problem.dimensionalize_fields(predicted),
            problem.dimensionalize_fields(exact),
        )
    plot_results(
        validation_output, problem, model, config["dtype"],
        (grid_size, grid_size), xy, predicted, exact, cell_residual, mesh,
    )
    return report, validation_output
