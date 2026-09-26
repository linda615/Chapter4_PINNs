from __future__ import annotations

def require_tf():
    try:
        import tensorflow as tf
        if not hasattr(tf, "keras"):
            raise ImportError
        return tf
    except ImportError as exc:
        raise RuntimeError("A complete TensorFlow 2 installation is required; run: python -m pip install -r requirements.txt") from exc


def build_model(width=48, trunk_depth=3, branch_depth=2, activation="tanh", lx=1.0, ly=1.0,
                output_scales=None, fourier_features=None, branch_layout="field",
                output_activation="linear"):
    tf = require_tf()
    if (fourier_features or {}).get("enabled", False):
        raise ValueError("Fourier input features are not included in this review release.")
    if branch_layout not in ("field", "scalar"):
        raise ValueError("branch_layout must be 'field' (four branches) or 'scalar' (eight branches)")
    if output_activation not in ("linear", "tanh"):
        raise ValueError("output_activation must be 'linear' or 'tanh'")
    output_scales = output_scales or {
        "w": [1.0], "beta": [1.0, 1.0], "moment": [1.0, 1.0, 1.0], "shear": [1.0, 1.0]
    }

    class PCRBNet(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.trunk = [tf.keras.layers.Dense(width, activation=activation) for _ in range(trunk_depth)]
            self.branch_layout = branch_layout
            branch_names = (
                ("w", "beta", "moment", "shear") if branch_layout == "field" else
                ("w", "beta_x", "beta_y", "M_xx", "M_yy", "M_xy", "Q_x", "Q_y")
            )
            self.branches = {
                name: [tf.keras.layers.Dense(width, activation=activation) for _ in range(branch_depth)]
                for name in branch_names
            }
            head_dims = {"w": 1, "beta": 2, "moment": 3, "shear": 2} if branch_layout == "field" else {
                name: 1 for name in branch_names
            }
            self.heads = {name: tf.keras.layers.Dense(dim) for name, dim in head_dims.items()}
            self.scales = {
                name: tf.constant(value, dtype=tf.as_dtype(tf.keras.backend.floatx()))[None, :]
                for name, value in output_scales.items()
            }

        def raw_outputs(self, xy, training=False):
            """Return unscaled configured-head outputs before boundary envelopes."""
            xn, yn = xy[:, 0:1]/lx, xy[:, 1:2]/ly
            z = tf.concat([2.0*xn-1.0, 2.0*yn-1.0], axis=1)
            for layer in self.trunk:
                z = layer(z, training=training)
            branch_out = {}
            for name, layers in self.branches.items():
                h = z
                for layer in layers:
                    h = h + layer(h, training=training)
                head = self.heads[name](h, training=training)
                branch_out[name] = head if output_activation == "linear" else tf.tanh(head)
            if self.branch_layout == "field":
                return {name: branch_out[name] for name in ("w", "beta", "moment", "shear")}
            return {
                "w": branch_out["w"],
                "beta": tf.concat([branch_out["beta_x"], branch_out["beta_y"]], axis=1),
                "moment": tf.concat([branch_out["M_xx"], branch_out["M_yy"], branch_out["M_xy"]], axis=1),
                "shear": tf.concat([branch_out["Q_x"], branch_out["Q_y"]], axis=1),
            }

        def call(self, xy, training=False):
            raw = self.raw_outputs(xy, training=training)
            out = {
                name: self.scales[name]*raw[name]
                for name in ("w", "beta", "moment", "shear")
            }
            xn, yn = xy[:, 0:1]/lx, xy[:, 1:2]/ly
            # Normalized envelopes have maximum one, so they impose essential
            # conditions without unintentionally shrinking the physical scale.
            out["w"] = 16.0*xn*(1.0-xn)*yn*(1.0-yn)*out["w"]
            beta = out["beta"]
            out["beta"] = tf.concat([
                4.0*yn*(1.0-yn)*beta[:, 0:1],
                4.0*xn*(1.0-xn)*beta[:, 1:2],
            ], axis=1)
            m = out["moment"]
            out["moment"] = tf.concat([
                4.0*xn*(1.0-xn)*m[:, 0:1],
                4.0*yn*(1.0-yn)*m[:, 1:2],
                m[:, 2:3],
            ], axis=1)
            return out

    return PCRBNet()
