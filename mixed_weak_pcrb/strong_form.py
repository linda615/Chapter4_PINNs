"""Eight-field first-order strong residuals and their fixed-quadrature loss.

The fields, scales and schedules are shared with the local weak formulation.
Only the domain constraint changes. Reference solutions are not used here.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .model import require_tf
from .quadrature import batched_cell_rule
from .trainer import PHYSICS_KEYS, Trainer


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def pointwise_jacobian(tape, field, xy, name="field"):
    """Per-point Jacobian for models without coupling between batch samples.

    PCRB-Net contains dense layers, activations and pointwise boundary maps;
    it has no batch normalization or other cross-sample operations. Therefore
    a componentwise vector-Jacobian product gives each sample's derivative.
    The supplied tape must be persistent and must have recorded ``field``.
    """
    tf = require_tf()
    if field.shape.rank != 2 or field.shape[-1] is None:
        raise ValueError(f"{name} must have shape (points, known components)")
    width = int(field.shape[-1])
    gradients = []
    for channel in range(width):
        selector = tf.broadcast_to(
            tf.one_hot(channel, width, dtype=field.dtype), tf.shape(field)
        )
        gradients.append(tape.gradient(
            field, xy, output_gradients=selector,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        ))
    return tf.stack(gradients, axis=1)


def pointwise_residuals(model, problem, xy, training=True):
    """Return residual groups of 2, 3, 2 and 1 components.

    Coordinates are those of ``problem``. In the dimensionless rectangular
    problem, rotations have their separate Lx/Ly recovery scales, while the
    moment/shear divergence uses factors (1, rho).
    """
    tf = require_tf()
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xy)
        out = model(xy, training=training)
        fields = tf.concat([out[k] for k in ("w", "beta", "moment", "shear")], axis=1)
    jac = pointwise_jacobian(tape, fields, xy, "eight fields")
    del tape
    gw, gb = jac[:, 0, :], jac[:, 1:3, :]
    gm, gq = jac[:, 3:6, :], jac[:, 6:8, :]
    mx, my = problem.divergence_factors
    return {
        "kinematic": out["beta"] - gw,
        "constitutive": out["moment"] - problem.constitutive_target_tf(gb, xy),
        "moment": tf.stack([
            out["shear"][:, 0] + mx * gm[:, 0, 0] + my * gm[:, 2, 1],
            out["shear"][:, 1] + mx * gm[:, 2, 0] + my * gm[:, 1, 1],
        ], axis=1),
        "equilibrium": (
            mx * gq[:, 0, 0] + my * gq[:, 1, 1] - problem.load_tf(xy)
        )[:, None],
    }


def strong_group_loss(model, problem, xy, weights, ncell, npoint, areas,
                      residual_scales, loss_weights, training=True):
    """Gauss-weighted mean of componentwise normalized squared residuals.

    The reduction first averages the components within each equation group,
    then integrates over cells and weights cells by area. No test functions
    or weak projections occur in this strong loss.
    """
    tf = require_tf()
    if residual_scales is None:
        raise ValueError("The strong-form control requires fixed prior residual scales")
    raw = pointwise_residuals(model, problem, xy, training)
    component = []
    for key in PHYSICS_KEYS:
        normalized = raw[key] / tf.constant(residual_scales[key], dtype=xy.dtype)[None, :]
        point_mean = tf.reshape(tf.reduce_mean(normalized ** 2, axis=1), (ncell, npoint))
        component.append(tf.reduce_sum(weights * point_mean, axis=1) / areas)
    per_cell_groups = tf.stack(component, axis=1)
    active = tf.stack([tf.cast(loss_weights[k], xy.dtype) for k in PHYSICS_KEYS])
    per_cell_total = tf.reduce_sum(per_cell_groups * active[None, :], axis=1)
    total_area = tf.reduce_sum(areas)
    diagnostics = tf.reduce_sum(areas[:, None] * per_cell_groups, axis=0) / total_area
    total = tf.reduce_sum(areas * per_cell_total) / total_area
    return total, tf.sqrt(per_cell_total), diagnostics


def prior_residual_metrics(model, problem, xy, residual_scales):
    """Common fixed-prior diagnostics; distinct from response-norm metrics.

    RMS averages points and components within each group. Raw tensor
    calculations retain model precision; division by the fixed NumPy scales
    uses float64, as in the reported strong/weak control evaluation.
    """
    residuals = pointwise_residuals(model, problem, xy, training=False)
    arrays = {key: value.numpy() for key, value in residuals.items()}
    report = {}
    for key, value in arrays.items():
        scales = np.asarray(residual_scales[key], dtype=np.float64)
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError(f"Invalid prior residual scales for {key}")
        normalized = value / scales
        report[key] = {
            "rms_component_mean": float(np.sqrt(np.mean(value ** 2))),
            "normalized_rms": float(np.sqrt(np.mean(normalized ** 2))),
            "normalized_max_absolute": float(np.max(np.abs(normalized))),
        }
    return report, arrays


class StrongTrainer(Trainer):
    """Single-run strong-form trainer with the PCRB-Net initialization/schedules."""

    def __init__(self, config):
        super().__init__(config)
        if self.residual_scales is None:
            raise ValueError("Enable fixed prior residual normalization for this control")
        tf = require_tf()
        td = tf.as_dtype(config["dtype"])
        # Only locations, weights and areas enter this loss; test order is inactive.
        rule = batched_cell_rule(self.cells, self.quadrature_order, 1)
        self.ncell, self.npoint = rule["xy"].shape[:2]
        self.xy = tf.constant(rule["xy"].reshape(-1, 2), td)
        self.integration_weights = tf.constant(rule["weights"], td)
        self.areas = tf.constant(rule["area"], td)

    def _compile_step(self):
        tf = require_tf()
        self._compiled_step = tf.function(
            self._step_impl, autograph=False, reduce_retracing=True,
            jit_compile=bool(self.cfg.get("training", {}).get("jit_compile", True)),
        )
        self._graph_needs_trace = True

    def _step_impl(self):
        tf = require_tf()
        with tf.GradientTape() as tape:
            loss, cell_rms, diagnostics = strong_group_loss(
                self.model, self.problem, self.xy, self.integration_weights,
                self.ncell, self.npoint, self.areas, self.residual_scales, self.loss_weights,
            )
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, cell_rms, diagnostics

    def train(self, epochs=None):
        epochs = int(self.cfg["training"]["epochs"] if epochs is None else epochs)
        if epochs < 1:
            raise ValueError("epochs must be positive")
        output = Path(self.cfg["output_dir"])
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents two runs from claiming the same directory.
        with (output / "run_state.json").open("x", encoding="utf-8") as stream:
            json.dump({"status": "initializing", "epochs": epochs}, stream)
        nodes = self.quadrature_nodes_per_step()
        config = dict(self.cfg)
        config["training"] = {**self.cfg["training"], "epochs": epochs}
        config["loss_form"] = "first_order_eight_field_strong"
        write_json(output / "resolved_config.json", config)
        write_json(output / "environment.json", config.get("execution", {}))
        write_json(output / "problem_metadata.json", {
            "loss_form": "first_order_eight_field_strong",
            "problem_class": type(self.problem).__name__,
            "coordinate_mapping": self.problem.coordinate_mapping,
            "network_output_scales": {k: np.asarray(v).tolist() for k, v in self.output_scales.items()},
            "residual_scales": {k: np.asarray(v).tolist() for k, v in self.residual_scales.items()},
            "output_scale_source": self.output_scale_source,
            "residual_scale_source": self.residual_scale_source,
            "training_points": {"type": "fixed composite Gauss nodes", "count": nodes,
                                "cells": self.ncell, "order": self.quadrature_order,
                                "test_modes_per_axis": None},
            "component_counts": [2, 3, 2, 1],
            "component_reduction": "mean of squared normalized components within each group",
            "spatial_reduction": "Gauss integral divided by domain area",
            "maximum_spatial_derivative_order": 1,
            "checkpoint_selection": "final epoch; no reference-based selection",
        })
        digest = hashlib.sha256()
        for value in self.model.get_weights():
            digest.update(value.tobytes())
        write_json(output / "initialization.json", {
            "seed": self.cfg.get("seed", 42), "initial_weights_sha256": digest.hexdigest(),
        })
        log_every = int(self.cfg["training"].get("log_every", 500))
        if log_every < 1:
            raise ValueError("log_every must be positive")
        start = time.perf_counter()
        try:
            with (output / "training_history.csv").open("x", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=self._history_fields())
                writer.writeheader()
                for epoch in range(1, epochs + 1):
                    self.apply_schedules(epoch)
                    loss, cell_rms, diagnostics = self.step()
                    if not np.isfinite(loss) or not np.all(np.isfinite(diagnostics)):
                        raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
                    writer.writerow(self._history_row(
                        epoch, loss, diagnostics, cell_rms, epoch * nodes, time.perf_counter() - start,
                    ))
                    if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                        stream.flush()
                        write_json(output / "run_state.json", {
                            "status": "training", "epoch": epoch, "epochs": epochs,
                            "elapsed_seconds": time.perf_counter() - start,
                        })
                        print(f"strong epoch={epoch} loss={loss:.8e} elapsed={time.perf_counter()-start:.1f}s", flush=True)
            elapsed = time.perf_counter() - start
            self.model.save_weights(str(output / "model.weights.h5"))
            write_json(output / "training_cost.json", {
                "epochs": epochs, "elapsed_seconds": elapsed,
                "cumulative_quadrature_nodes": epochs * nodes, "nodes_per_step": nodes,
                "parameters": int(self.model.count_params()),
                "timing_scope": "training loop including first graph/XLA compilation and logging; excluding checkpoint save and validation",
            })
            write_json(output / "run_state.json", {"status": "complete", "epoch": epochs, "epochs": epochs})
        except BaseException as exc:
            write_json(output / "run_state.json", {"status": "failed", "error": repr(exc)})
            raise
        return output
