from __future__ import annotations

from .model import require_tf
from .quadrature import batched_cell_rule


def batch_jacobian(tape, field, xy, name):
    """Return d(field[n, i])/d(xy[n, j]) without cross-sample terms."""
    tf = require_tf()
    jac = tape.batch_jacobian(
        field,
        xy,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
        experimental_use_pfor=True,
    )
    if jac is None:
        raise RuntimeError(f"Automatic differentiation failed for {name}; check that the model output depends on xy.")
    return jac


def batched_cell_residuals(model, problem, cells, quadrature_order, test_order, dtype="float64", training=True):
    """Evaluate all cells with one network call and one Jacobian pair."""
    tf = require_tf()
    rule = batched_cell_rule(cells, quadrature_order, test_order)
    td = tf.float64 if dtype == "float64" else tf.float32
    ncell, npoint = rule["xy"].shape[:2]
    xy = tf.convert_to_tensor(rule["xy"].reshape(-1, 2), dtype=td)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xy)
        out = model(xy, training=training)
        w, beta, moment, shear = out["w"], out["beta"], out["moment"], out["shear"]
    gw = batch_jacobian(tape, w, xy, "w")[:, 0, :]
    gb = batch_jacobian(tape, beta, xy, "beta")
    del tape
    beta = tf.reshape(beta, (ncell, npoint, 2))
    moment = tf.reshape(moment, (ncell, npoint, 3))
    shear = tf.reshape(shear, (ncell, npoint, 2))
    gw = tf.reshape(gw, (ncell, npoint, 2))
    gb = tf.reshape(gb, (ncell, npoint, 2, 2))
    weights = tf.convert_to_tensor(rule["weights"], td)
    phi = tf.convert_to_tensor(rule["phi"], td)
    grad_phi = tf.convert_to_tensor(rule["grad_phi"], td)
    norm = tf.sqrt(tf.reduce_sum(phi**2*weights[:, None, :], axis=2)+tf.cast(1e-14, td))
    scale = norm*tf.sqrt(tf.convert_to_tensor(rule["area"], td))[:, None]

    def project_scalar(value):
        return tf.reduce_sum(phi*weights[:, None, :]*value[:, None, :], axis=2)/scale

    kin = tf.stack([project_scalar(beta[:, :, i]-gw[:, :, i]) for i in range(2)], axis=2)
    target = tf.reshape(problem.constitutive_target_tf(
        tf.reshape(gb, (-1, 2, 2)), xy
    ), (ncell, npoint, 3))
    constitutive = tf.stack([project_scalar(moment[:, :, i]-target[:, :, i]) for i in range(3)], axis=2)

    # Weak Q+div(M)=0; M=[Mxx, Myy, Mxy]. Bubble tests make cell boundary terms vanish.
    wx = weights[:, None, :]
    metric_x, metric_y = problem.divergence_factors
    rq_x = tf.reduce_sum(wx*(phi*shear[:, None, :, 0] - metric_x*grad_phi[:, :, :, 0]*moment[:, None, :, 0] - metric_y*grad_phi[:, :, :, 1]*moment[:, None, :, 2]), axis=2)/scale
    rq_y = tf.reduce_sum(wx*(phi*shear[:, None, :, 1] - metric_x*grad_phi[:, :, :, 0]*moment[:, None, :, 2] - metric_y*grad_phi[:, :, :, 1]*moment[:, None, :, 1]), axis=2)/scale
    moment_balance = tf.stack([rq_x, rq_y], axis=2)
    load = tf.reshape(problem.load_tf(xy), (ncell, npoint))
    equilibrium = tf.reduce_sum(wx*(-metric_x*grad_phi[:, :, :, 0]*shear[:, None, :, 0]-metric_y*grad_phi[:, :, :, 1]*shear[:, None, :, 1]-phi*load[:, None, :]), axis=2)/scale
    return {"kinematic": kin, "constitutive": constitutive, "moment": moment_balance, "equilibrium": equilibrium}


def residual_loss_per_cell(residuals, weights):
    tf = require_tf()
    return sum(tf.cast(weights[name], value.dtype)*tf.reduce_mean(value**2, axis=tf.range(1, tf.rank(value))) for name, value in residuals.items())


def normalize_residuals(residuals, residual_scales):
    """Nondimensionalize projected residual components with fixed prior scales."""
    if residual_scales is None:
        return residuals
    tf = require_tf()
    normalized = {}
    for name, value in residuals.items():
        if name not in residual_scales:
            raise ValueError(f"missing normalization scale for residual '{name}'")
        scale = tf.convert_to_tensor(residual_scales[name], dtype=value.dtype)
        if value.shape.rank == 3:
            scale = tf.reshape(scale, (1, 1, -1))
        elif value.shape.rank == 2:
            scale = tf.reshape(scale, (1, -1))
        else:
            raise ValueError(f"unsupported residual rank for '{name}': {value.shape.rank}")
        normalized[name] = value/scale
    return normalized


def fixed_group_loss(model, problem, cells, order, test_order, weights, dtype,
                     residual_scales=None):
    """Physical weak-form loss evaluated with exactly one fixed Gauss order."""
    tf = require_tf()
    raw_residuals = batched_cell_residuals(model, problem, cells, order, test_order, dtype)
    residuals = normalize_residuals(raw_residuals, residual_scales)
    component_losses = {
        key: tf.reduce_mean(value**2, axis=tf.range(1, tf.rank(value)))
        for key, value in residuals.items()
    }
    physical = residual_loss_per_cell(residuals, weights)
    diagnostics = tf.stack([
        component_losses["kinematic"], component_losses["constitutive"],
        component_losses["moment"], component_losses["equilibrium"],
    ], axis=1)
    return physical, diagnostics, raw_residuals
