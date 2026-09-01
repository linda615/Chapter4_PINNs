from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .model import build_model, require_tf
from .problem import build_problem, resolve_output_scales, resolve_residual_scales
from .quadrature import RectMesh
from .weak_form import fixed_group_loss


PHYSICS_KEYS = ("kinematic", "constitutive", "moment", "equilibrium")


class Trainer:
    """Train PCRS-Net with one fixed tensor-product Gauss rule."""

    def __init__(self, config):
        self.cfg = config
        weak = config["weak_form"]
        self._validate_schedule(weak.get("weight_schedule"), "weak_form.weight_schedule")
        self._validate_schedule(
            config["training"].get("learning_rate_schedule"),
            "training.learning_rate_schedule",
        )
        self._validate_transition(weak.get("weight_transition"), "weak_form.weight_transition")
        self._validate_transition(
            config["training"].get("learning_rate_transition"),
            "training.learning_rate_transition",
        )

        tf = require_tf()
        tf.keras.backend.set_floatx(config.get("dtype", "float64"))
        tf.keras.utils.set_random_seed(config.get("seed", 42))
        dc, nc = config["domain"], config["network"]
        self.problem = build_problem(config)
        self.mesh = RectMesh(dc["lx"], dc["ly"], dc["nx"], dc["ny"])
        self.cells = self.mesh.cells.copy()
        self.quadrature_order = int(weak["quadrature_order"])
        self.test_order = int(weak["test_order"])
        if self.quadrature_order != 8:
            raise ValueError("This review release trains with fixed quadrature G=8.")

        self.output_scales, self.output_scale_source = resolve_output_scales(
            config, self.problem
        )
        self.residual_scales, self.residual_scale_source = resolve_residual_scales(
            config, self.problem
        )
        if self.residual_scales is None:
            raise ValueError("Residual normalization must be enabled in this review release.")
        self.model = build_model(
            **nc,
            lx=dc["lx"],
            ly=dc["ly"],
            output_scales=self.output_scales,
        )
        td = tf.float64 if config.get("dtype", "float64") == "float64" else tf.float32
        self.model(tf.zeros((1, 2), dtype=td), training=False)

        configured_weights = weak["weights"]
        missing = [name for name in PHYSICS_KEYS if name not in configured_weights]
        extra = [name for name in configured_weights if name not in PHYSICS_KEYS]
        if missing or extra:
            raise ValueError(
                f"weak_form.weights must contain exactly {PHYSICS_KEYS}; "
                f"missing={missing}, extra={extra}"
            )
        self.loss_weights = {
            name: tf.Variable(
                configured_weights[name],
                dtype=td,
                trainable=False,
                name=f"loss_weight_{name}",
            )
            for name in PHYSICS_KEYS
        }
        self.optimizer = tf.keras.optimizers.Adam(config["training"]["learning_rate"])
        self._active_weight_stage = None
        self._active_lr_stage = None
        self._compile_step()

    def _step_impl(self):
        tf = require_tf()
        with tf.GradientTape() as tape:
            values, diagnostics, _ = fixed_group_loss(
                self.model,
                self.problem,
                self.cells,
                self.quadrature_order,
                self.test_order,
                self.loss_weights,
                self.cfg["dtype"],
                self.residual_scales,
            )
            areas = (
                (self.cells[:, 1] - self.cells[:, 0])
                * (self.cells[:, 3] - self.cells[:, 2])
            )
            area_tensor = tf.constant(areas, dtype=values.dtype)
            total_area = tf.reduce_sum(area_tensor)
            loss = tf.reduce_sum(area_tensor * values) / total_area
            diagnostic_vector = (
                tf.reduce_sum(area_tensor[:, None] * diagnostics, axis=0) / total_area
            )
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, tf.sqrt(values), diagnostic_vector

    def _compile_step(self):
        tf = require_tf()
        self._compiled_step = tf.function(
            self._step_impl,
            autograph=False,
            reduce_retracing=True,
        )
        self._graph_needs_trace = True

    def step(self):
        if self._graph_needs_trace:
            print(
                "Tracing TensorFlow training graph for fixed quadrature "
                f"G={self.quadrature_order} ..."
            )
        loss, residuals, diagnostics = self._compiled_step()
        self._graph_needs_trace = False
        return float(loss.numpy()), residuals.numpy(), diagnostics.numpy()

    @staticmethod
    def _history_fields():
        return [
            "epoch",
            "total_loss",
            "kinematic_loss",
            "constitutive_loss",
            "moment_loss",
            "equilibrium_loss",
            "learning_rate",
            "weight_kinematic",
            "weight_constitutive",
            "weight_moment",
            "weight_equilibrium",
            "max_local_residual",
            "cell_count",
            "quadrature_order",
            "quadrature_nodes_per_step",
            "cumulative_quadrature_nodes",
            "elapsed_seconds",
        ]

    def _history_row(
        self,
        epoch,
        loss,
        diagnostics,
        residuals,
        cumulative_nodes,
        elapsed_seconds,
    ):
        return {
            "epoch": epoch,
            "total_loss": loss,
            "kinematic_loss": float(diagnostics[0]),
            "constitutive_loss": float(diagnostics[1]),
            "moment_loss": float(diagnostics[2]),
            "equilibrium_loss": float(diagnostics[3]),
            "learning_rate": float(self.optimizer.learning_rate.numpy()),
            **{
                f"weight_{name}": float(value.numpy())
                for name, value in self.loss_weights.items()
            },
            "max_local_residual": float(np.max(residuals)),
            "cell_count": len(self.cells),
            "quadrature_order": self.quadrature_order,
            "quadrature_nodes_per_step": self.quadrature_nodes_per_step(),
            "cumulative_quadrature_nodes": int(cumulative_nodes),
            "elapsed_seconds": float(elapsed_seconds),
        }

    def quadrature_nodes_per_step(self):
        return int(len(self.cells) * self.quadrature_order**2)

    @staticmethod
    def _validate_schedule(schedule, name):
        previous_end = None
        for index, stage in enumerate(schedule or []):
            start, end = int(stage["start"]), int(stage["end"])
            if start > end:
                raise ValueError(f"{name}[{index}] has start > end: {start} > {end}")
            if previous_end is not None and start <= previous_end:
                raise ValueError(
                    f"{name} stages overlap at epoch {start}; "
                    f"previous stage ends at {previous_end}"
                )
            previous_end = end

    @staticmethod
    def _find_stage(schedule, epoch):
        for index, stage in enumerate(schedule or []):
            if int(stage["start"]) <= epoch <= int(stage["end"]):
                return index, stage
        return None, None

    @staticmethod
    def _validate_transition(config, name):
        if not config:
            return
        if config.get("type", "cosine") not in ("linear", "cosine"):
            raise ValueError(f"{name}.type must be 'linear' or 'cosine'")
        if int(config.get("epochs", 0)) < 1:
            raise ValueError(f"{name}.epochs must be positive")

    @staticmethod
    def _transition_fraction(epoch, stage_start, config):
        if not config:
            return 1.0
        duration = int(config["epochs"])
        progress = np.clip((epoch - (int(stage_start) - 1)) / duration, 0.0, 1.0)
        if config.get("type", "cosine") == "cosine":
            return float(0.5 * (1.0 - np.cos(np.pi * progress)))
        return float(progress)

    @classmethod
    def _interpolated_stage_values(cls, schedule, index, epoch, transition, keys):
        stage = schedule[index]
        if index == 0 or not transition:
            return {key: float(stage[key]) for key in keys if key in stage}
        alpha = cls._transition_fraction(epoch, stage["start"], transition)
        previous = schedule[index - 1]
        return {
            key: (1.0 - alpha) * float(previous.get(key, stage[key]))
            + alpha * float(stage[key])
            for key in keys
            if key in stage
        }

    def apply_schedules(self, epoch):
        weight_schedule = self.cfg["weak_form"].get("weight_schedule") or []
        weight_index, weight_stage = self._find_stage(weight_schedule, epoch)
        if weight_stage is not None:
            transition = self.cfg["weak_form"].get("weight_transition")
            values = self._interpolated_stage_values(
                weight_schedule,
                weight_index,
                epoch,
                transition,
                self.loss_weights,
            )
            for name, value in values.items():
                self.loss_weights[name].assign(value)
            if weight_index != self._active_weight_stage:
                self._active_weight_stage = weight_index
                print(
                    f"Loss-weight stage {weight_index + 1}: epochs "
                    f"{weight_stage['start']}-{weight_stage['end']}"
                )

        lr_schedule = self.cfg["training"].get("learning_rate_schedule") or []
        lr_index, lr_stage = self._find_stage(lr_schedule, epoch)
        if lr_stage is not None:
            transition = self.cfg["training"].get("learning_rate_transition")
            value = self._interpolated_stage_values(
                lr_schedule,
                lr_index,
                epoch,
                transition,
                ("value",),
            )["value"]
            self.optimizer.learning_rate.assign(value)
            if lr_index != self._active_lr_stage:
                self._active_lr_stage = lr_index
                print(
                    f"Learning-rate stage {lr_index + 1}: epochs "
                    f"{lr_stage['start']}-{lr_stage['end']}"
                )

    def train(self, epochs=None):
        epochs = int(epochs or self.cfg["training"]["epochs"])
        output = Path(self.cfg["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "resolved_config.json").write_text(
            json.dumps(self.cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        metadata = {
            "problem_class": type(self.problem).__name__,
            "coordinate_mapping": self.problem.coordinate_mapping,
            "quadrature": {
                "mode": "fixed",
                "order": self.quadrature_order,
                "test_modes_per_direction": self.test_order,
            },
            "output_scale_source": self.output_scale_source,
            "network_output_scales": {
                name: np.asarray(value).tolist()
                for name, value in self.output_scales.items()
            },
            "residual_normalization_enabled": self.residual_scales is not None,
            "residual_scale_source": self.residual_scale_source,
            "residual_scales": None
            if self.residual_scales is None
            else {
                name: np.asarray(value).tolist()
                for name, value in self.residual_scales.items()
            },
        }
        if hasattr(self.problem, "physical_field_scales"):
            metadata["dimensional_recovery_scales"] = {
                name: np.asarray(value).tolist()
                for name, value in self.problem.physical_field_scales.items()
            }
        (output / "problem_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(
            f"Quadrature mode: fixed G={self.quadrature_order}; "
            f"test modes per direction={self.test_order}"
        )
        print(
            "Residual normalization: "
            + (
                f"enabled ({self.residual_scale_source})"
                if self.residual_scales is not None
                else "disabled"
            )
        )

        history_path = output / "training_history.csv"
        log_every = int(self.cfg["training"].get("log_every", 50))
        nodes_per_step = self.quadrature_nodes_per_step()
        cumulative_nodes = 0
        start = time.perf_counter()
        with history_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=self._history_fields())
            writer.writeheader()
            for epoch in range(1, epochs + 1):
                self.apply_schedules(epoch)
                loss, residuals, diagnostics = self.step()
                cumulative_nodes += nodes_per_step
                writer.writerow(
                    self._history_row(
                        epoch,
                        loss,
                        diagnostics,
                        residuals,
                        cumulative_nodes,
                        time.perf_counter() - start,
                    )
                )
                if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                    stream.flush()
                    print(
                        f"epoch={epoch:6d} loss={loss:.6e} "
                        f"lr={float(self.optimizer.learning_rate.numpy()):.3e} "
                        f"max_local_residual={np.max(residuals):.3e}"
                    )

        elapsed = time.perf_counter() - start
        self.model.save_weights(output / "model.weights.h5")
        (output / "training_cost.json").write_text(
            json.dumps(
                {
                    "epochs": epochs,
                    "elapsed_seconds": elapsed,
                    "cumulative_quadrature_nodes": cumulative_nodes,
                    "nodes_per_step": nodes_per_step,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Training artifacts written to: {output}")
        return output
