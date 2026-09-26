"""Independent checks of rectangular metrics, weighted loss and nested gradients."""

import os
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf

from mixed_weak_pcrb.problem import LocalGaussianPlate
from mixed_weak_pcrb.strong_form import (
    PHYSICS_KEYS, pointwise_residuals, prior_residual_metrics, strong_group_loss,
)


class PolynomialFields:
    def __init__(self, gain=1.0):
        self.gain = tf.Variable(gain, dtype=tf.float64)

    def __call__(self, xy, training=False):
        x, y = xy[:, 0], xy[:, 1]
        fields = {
            "w": (x ** 3 + 2 * x * y + 3 * y ** 2)[:, None],
            "beta": tf.stack([x ** 2 + 2 * y, 3 * x + y ** 2], axis=1),
            "moment": tf.stack([x ** 2 * y, x * y ** 2, x ** 2 + 2 * y ** 2], axis=1),
            "shear": tf.stack([x * y, x ** 2 * y], axis=1),
        }
        return {key: self.gain * value for key, value in fields.items()}


class StrongFormTests(unittest.TestCase):
    def setUp(self):
        self.problem = LocalGaussianPlate()
        self.xy = tf.constant([[.13, .29], [.57, .84], [.91, .31], [.43, .67]], tf.float64)
        self.scales = {
            "kinematic": np.array([2., 3.]), "constitutive": np.array([4., 5., 6.]),
            "moment": np.array([7., 8.]), "equilibrium": np.array([9.]),
        }
        self.loss_weights = {k: tf.constant(v, tf.float64)
                             for k, v in zip(PHYSICS_KEYS, [1., .7, .3, .2])}
        self.weights = tf.constant([[.10, .15], [.20, .55]], tf.float64)
        self.areas = tf.constant([.25, .75], tf.float64)

    def expected_polynomial(self):
        x, y = self.xy.numpy().T
        rho = self.problem.aspect_ratio
        self.assertEqual(rho, 2.)
        return {
            "kinematic": np.c_[-2 * x ** 2, x + y ** 2 - 6 * y],
            "constitutive": np.c_[
                x ** 2 * y + 2 * x + .6 * rho ** 2 * y,
                x * y ** 2 + 2 * rho ** 2 * y + .6 * x,
                x ** 2 + 2 * y ** 2 + 1.75 * rho,
            ],
            "moment": np.c_[3 * x * y + 4 * rho * y, x ** 2 * y + 2 * x + 2 * rho * x * y],
            "equilibrium": (y + rho * x ** 2 - self.problem.load_numpy(self.xy.numpy()))[:, None],
        }

    def loss(self, model):
        return strong_group_loss(
            model, self.problem, self.xy, self.weights, 2, 2, self.areas,
            self.scales, self.loss_weights,
        )[0]

    def test_rectangular_polynomial_residuals(self):
        actual = pointwise_residuals(PolynomialFields(), self.problem, self.xy, False)
        for key, expected in self.expected_polynomial().items():
            np.testing.assert_allclose(actual[key].numpy(), expected, atol=1e-12, rtol=1e-12)

    def test_unequal_area_and_component_reduction(self):
        manual = 0.0
        for key, values in self.expected_polynomial().items():
            per_point = np.mean((values / self.scales[key]) ** 2, axis=1)
            manual += float(self.loss_weights[key]) * np.sum(
                self.weights.numpy().ravel() * per_point
            ) / float(tf.reduce_sum(self.areas))
        self.assertAlmostEqual(float(self.loss(PolynomialFields())), manual, places=12)

    def test_parameter_gradient_through_spatial_derivatives(self):
        model = PolynomialFields(1.3)
        with tf.GradientTape() as tape:
            value = self.loss(model)
        derivative = float(tape.gradient(value, model.gain))
        step = 1e-5
        model.gain.assign(1.3 + step)
        plus = float(self.loss(model))
        model.gain.assign(1.3 - step)
        minus = float(self.loss(model))
        finite_difference = (plus - minus) / (2 * step)
        self.assertLess(abs(derivative - finite_difference) / max(abs(derivative), 1e-12), 1e-8)

    def test_common_prior_rms_uses_all_components(self):
        report, raw = prior_residual_metrics(PolynomialFields(), self.problem, self.xy, self.scales)
        for key in PHYSICS_KEYS:
            expected = np.sqrt(np.mean((self.expected_polynomial()[key] / self.scales[key]) ** 2))
            self.assertAlmostEqual(report[key]["normalized_rms"], expected, places=12)
            self.assertEqual(raw[key].shape[0], 4)


if __name__ == "__main__":
    unittest.main()
