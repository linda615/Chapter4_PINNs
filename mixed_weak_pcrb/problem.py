from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import numpy as np


OUTPUT_SCALE_COMPONENTS = (
    "w", "beta_x", "beta_y", "M_xx", "M_yy", "M_xy", "Q_x", "Q_y",
)

RESIDUAL_SCALE_COMPONENTS = {
    "kinematic": 2,
    "constitutive": 3,
    "moment": 2,
    "equilibrium": 1,
}


def resolve_output_scales(config, problem) -> tuple[dict[str, np.ndarray], str]:
    """Return grouped network scales, honoring an optional explicit config override."""
    explicit = config.get("output_scale")
    if explicit is None:
        return {
            name: np.asarray(value, dtype=float).copy()
            for name, value in problem.output_scales.items()
        }, getattr(problem, "output_scale_source", "problem-specific a-priori scale")
    missing = [name for name in OUTPUT_SCALE_COMPONENTS if name not in explicit]
    extra = [name for name in explicit if name not in OUTPUT_SCALE_COMPONENTS]
    if missing or extra:
        raise ValueError(
            "output_scale must define exactly "
            f"{OUTPUT_SCALE_COMPONENTS}; missing={missing}, extra={extra}"
        )
    values = np.asarray([explicit[name] for name in OUTPUT_SCALE_COMPONENTS], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("all explicit output_scale values must be finite and positive")
    return {
        "w": values[0:1],
        "beta": values[1:3],
        "moment": values[3:6],
        "shear": values[6:8],
    }, "explicit dimensionless component scales from config.output_scale"


def resolve_residual_scales(config, problem) -> tuple[dict[str, np.ndarray] | None, str]:
    """Return fixed physics-based scales for equation-wise loss normalization.

    The scales deliberately come from ``problem.residual_scales`` rather than
    ``config.output_scale``.  An explicit network-output scale may be selected
    from a reference solution for a benchmark, whereas residual normalization
    must remain available when the solution is unknown.
    """
    settings = config.get("weak_form", {}).get("residual_normalization", {})
    if not settings.get("enabled", False):
        return None, "disabled"
    source = settings.get("source", "problem_prior")
    if source != "problem_prior":
        raise ValueError("residual_normalization.source must be 'problem_prior'")
    minimum = float(settings.get("minimum_scale", 1.0e-10))
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("residual_normalization.minimum_scale must be finite and positive")
    if not hasattr(problem, "residual_scales"):
        raise ValueError(f"{type(problem).__name__} does not define prior residual scales")
    scales = {}
    for name, count in RESIDUAL_SCALE_COMPONENTS.items():
        value = np.asarray(problem.residual_scales[name], dtype=float).reshape(-1)
        if value.size != count:
            raise ValueError(
                f"residual scale '{name}' must contain {count} component(s), got {value.size}"
            )
        if np.any(~np.isfinite(value)) or np.any(value <= 0.0):
            raise ValueError(f"residual scale '{name}' must be finite and positive")
        scales[name] = np.maximum(value, minimum)
    return scales, getattr(
        problem,
        "residual_scale_source",
        "problem-specific a-priori physical response scales",
    )


@dataclass(frozen=True)
class NavierPlate:
    """Simply-supported rectangular plate with q=q0*sin(pi*x/Lx)*sin(pi*y/Ly)."""

    lx: float = 1.0
    ly: float = 1.0
    rigidity: float = 1.0
    poisson: float = 0.3
    load_amplitude: float = 1.0
    output_scale_margin: float = 2.0

    @property
    def ax(self) -> float:
        return np.pi / self.lx

    @property
    def ay(self) -> float:
        return np.pi / self.ly

    @property
    def amplitude(self) -> float:
        return self.load_amplitude / (self.rigidity * (self.ax**2 + self.ay**2) ** 2)

    @property
    def output_scales(self) -> dict[str, np.ndarray]:
        """Physics-based peak scales for dimensionless network branch outputs.

        These follow from the known load wave numbers and plate parameters, not
        from sampled solution labels. They equal the analytical peak magnitudes
        for the single Navier mode used by this verification problem.
        """
        a, ax, ay, d, nu = abs(self.amplitude), self.ax, self.ay, self.rigidity, self.poisson
        margin = self.output_scale_margin
        return {
            "w": margin*np.asarray([a]),
            "beta": margin*np.asarray([a*ax, a*ay]),
            "moment": margin*np.asarray([
                d*a*(ax**2+nu*ay**2),
                d*a*(ay**2+nu*ax**2),
                d*a*(1.0-nu)*ax*ay,
            ]),
            "shear": margin*np.asarray([
                d*a*ax*(ax**2+ay**2),
                d*a*ay*(ax**2+ay**2),
            ]),
        }

    @property
    def residual_scales(self) -> dict[str, np.ndarray]:
        """A-priori RMS equation scales derived from the prescribed sine load."""
        response = {
            # Every first-mode field is a product of sine/cosine factors and
            # therefore has RMS equal to one half of its peak magnitude.
            name: np.asarray(value, dtype=float)/(2.0*self.output_scale_margin)
            for name, value in self.output_scales.items()
        }
        return {
            "kinematic": response["beta"],
            "constitutive": response["moment"],
            "moment": response["shear"],
            "equilibrium": np.asarray([0.5*abs(self.load_amplitude)]),
        }

    @property
    def residual_scale_source(self) -> str:
        return "RMS scales from prescribed sine load, rigidity and wave numbers"

    @property
    def divergence_factors(self) -> tuple[float, float]:
        """Metric factors multiplying x- and y-derivatives."""
        return 1.0, 1.0

    @property
    def reference_label(self) -> str:
        return "解析解"

    @property
    def coordinate_labels(self) -> tuple[str, str]:
        return r"$x$", r"$y$"

    @property
    def coordinate_mapping(self) -> dict:
        return {
            "type": "physical",
            "computational_domain": [0.0, self.lx, 0.0, self.ly],
        }

    def exact_numpy(self, xy: np.ndarray) -> dict[str, np.ndarray]:
        x, y = xy[:, 0], xy[:, 1]
        ax, ay, a = self.ax, self.ay, self.amplitude
        sx, sy, cx, cy = np.sin(ax*x), np.sin(ay*y), np.cos(ax*x), np.cos(ay*y)
        w = a*sx*sy
        bx, by = a*ax*cx*sy, a*ay*sx*cy
        wxx, wyy, wxy = -a*ax**2*sx*sy, -a*ay**2*sx*sy, a*ax*ay*cx*cy
        d, nu = self.rigidity, self.poisson
        mxx = -d*(wxx + nu*wyy)
        myy = -d*(wyy + nu*wxx)
        mxy = -d*(1.0-nu)*wxy
        qx = -d*a*ax*(ax**2+ay**2)*cx*sy
        qy = -d*a*ay*(ax**2+ay**2)*sx*cy
        load = self.load_amplitude*sx*sy
        return {"w": w, "beta": np.c_[bx, by], "moment": np.c_[mxx, myy, mxy], "shear": np.c_[qx, qy], "load": load}

    def load_numpy(self, xy: np.ndarray) -> np.ndarray:
        x, y = xy[:, 0], xy[:, 1]
        return self.load_amplitude*np.sin(self.ax*x)*np.sin(self.ay*y)

    def load_tf(self, xy):
        import tensorflow as tf
        x, y = xy[:, 0], xy[:, 1]
        return self.load_amplitude * tf.sin(np.pi*x/self.lx) * tf.sin(np.pi*y/self.ly)

    def rigidity_tf(self, xy):
        import tensorflow as tf
        return tf.ones_like(xy[:, 0]) * self.rigidity

    def constitutive_target_tf(self, beta_gradient, xy):
        """Return isotropic bending moments for the supplied rotation gradient."""
        import tensorflow as tf
        exx, eyy = beta_gradient[:, 0, 0], beta_gradient[:, 1, 1]
        exy = 0.5*(beta_gradient[:, 0, 1]+beta_gradient[:, 1, 0])
        d, nu = self.rigidity_tf(xy), self.poisson
        return tf.stack([
            -d*(exx+nu*eyy),
            -d*(eyy+nu*exx),
            -d*(1.0-nu)*exy,
        ], axis=1)


@dataclass(frozen=True)
class LocalGaussianPlate:
    """Dimensionless simply-supported plate under a finite-width local load.

    The physical rectangle ``[0, physical_lx] x [0, physical_ly]`` is mapped to
    ``(xi, eta) in [0, 1]^2``. Network outputs are dimensionless. The physical
    aspect ratio is retained by derivative metric factors, so this mapping does
    not turn a physical rectangle into a square plate.
    """

    physical_lx: float = 2.0
    physical_ly: float = 1.0
    rigidity: float = 1.0
    poisson: float = 0.3
    load_amplitude: float = 1.0
    load_center_x: float = 1.0
    load_center_y: float = 0.5
    load_radius_x: float = 0.16
    load_radius_y: float = 0.16
    reference_terms: int = 61
    coefficient_quadrature_order: int = 256
    output_scale_margin: float = 1.5

    def __post_init__(self):
        if self.physical_lx <= 0.0 or self.physical_ly <= 0.0:
            raise ValueError("physical plate lengths must be positive")
        if self.rigidity <= 0.0 or self.load_amplitude <= 0.0:
            raise ValueError("rigidity and load_amplitude must be positive")
        if self.load_radius_x <= 0.0 or self.load_radius_y <= 0.0:
            raise ValueError("local-load radii must be positive")
        if not (0.0 <= self.load_center_x <= self.physical_lx):
            raise ValueError("load_center_x lies outside the physical plate")
        if not (0.0 <= self.load_center_y <= self.physical_ly):
            raise ValueError("load_center_y lies outside the physical plate")
        if self.reference_terms < 1 or self.coefficient_quadrature_order < 2:
            raise ValueError("reference series and coefficient quadrature orders are invalid")

    @property
    def lx(self) -> float:
        """Computational-domain length in xi."""
        return 1.0

    @property
    def ly(self) -> float:
        """Computational-domain length in eta."""
        return 1.0

    @property
    def aspect_ratio(self) -> float:
        return self.physical_lx/self.physical_ly

    @property
    def normalized_center(self) -> tuple[float, float]:
        return self.load_center_x/self.physical_lx, self.load_center_y/self.physical_ly

    @property
    def normalized_radii(self) -> tuple[float, float]:
        return self.load_radius_x/self.physical_lx, self.load_radius_y/self.physical_ly

    @property
    def divergence_factors(self) -> tuple[float, float]:
        # Lc=Lx gives d/dx -> d/dxi and d/dy -> (Lx/Ly)d/deta
        return 1.0, self.aspect_ratio

    @property
    def reference_label(self) -> str:
        return "Navier级数参考解"

    @property
    def coordinate_labels(self) -> tuple[str, str]:
        return r"$\xi$", r"$\eta$"

    @property
    def coordinate_mapping(self) -> dict:
        return {
            "type": "dimensionless_unit_square",
            "physical_domain": [0.0, self.physical_lx, 0.0, self.physical_ly],
            "computational_domain": [0.0, 1.0, 0.0, 1.0],
            "mapping": "xi=x/Lx, eta=y/Ly",
            "aspect_ratio_Lx_over_Ly": self.aspect_ratio,
        }

    @property
    def physical_field_scales(self) -> dict[str, np.ndarray]:
        """Scales that recover dimensional fields from network outputs."""
        p, a, b, d = self.load_amplitude, self.physical_lx, self.physical_ly, self.rigidity
        wc = p*a**4/d
        return {
            "w": np.asarray([wc]),
            "beta": np.asarray([wc/a, wc/b]),
            "moment": np.asarray([p*a**2, p*a**2, p*a**2]),
            "shear": np.asarray([p*a, p*a]),
            "load": np.asarray([p]),
        }

    def dimensionalize_fields(self, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        scales = self.physical_field_scales
        result = {
            "w": np.asarray(fields["w"])*scales["w"][0],
            "beta": np.asarray(fields["beta"])*scales["beta"][None, :],
            "moment": np.asarray(fields["moment"])*scales["moment"][None, :],
            "shear": np.asarray(fields["shear"])*scales["shear"][None, :],
        }
        if "load" in fields:
            result["load"] = np.asarray(fields["load"])*scales["load"][0]
        return result

    def computational_to_physical(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return np.c_[xy[:, 0]*self.physical_lx, xy[:, 1]*self.physical_ly]

    @cached_property
    def first_load_coefficient(self) -> float:
        """First sine coefficient of the known load, without solving the plate."""
        from numpy.polynomial.legendre import leggauss

        z, wz = leggauss(self.coefficient_quadrature_order)
        points, weights = 0.5*(z+1.0), 0.5*wz
        cx, cy = self.normalized_center
        rx, ry = self.normalized_radii
        gx = np.exp(-((points-cx)/rx)**2)
        gy = np.exp(-((points-cy)/ry)**2)
        sine = np.sin(np.pi*points)
        return float(4.0*np.sum(weights*gx*sine)*np.sum(weights*gy*sine))

    @cached_property
    def load_rms(self) -> float:
        """RMS of the prescribed dimensionless Gaussian load on the unit square."""
        from numpy.polynomial.legendre import leggauss

        z, wz = leggauss(self.coefficient_quadrature_order)
        points, weights = 0.5*(z+1.0), 0.5*wz
        cx, cy = self.normalized_center
        rx, ry = self.normalized_radii
        gx2 = np.exp(-2.0*((points-cx)/rx)**2)
        gy2 = np.exp(-2.0*((points-cy)/ry)**2)
        return float(np.sqrt(np.sum(weights*gx2)*np.sum(weights*gy2)))

    @cached_property
    def modal_data(self) -> dict[str, np.ndarray]:
        """Navier coefficients computed independently with high-order Gauss rules."""
        from numpy.polynomial.legendre import leggauss

        z, wz = leggauss(self.coefficient_quadrature_order)
        points, weights = 0.5*(z+1.0), 0.5*wz
        modes = np.arange(1, self.reference_terms+1, dtype=float)
        wave = np.pi*modes
        cx, cy = self.normalized_center
        rx, ry = self.normalized_radii
        gx = np.exp(-((points-cx)/rx)**2)
        gy = np.exp(-((points-cy)/ry)**2)
        sine = np.sin(np.outer(points, wave))
        ix = (weights*gx)@sine
        iy = (weights*gy)@sine
        load_coeff = 4.0*np.outer(ix, iy)  # normalized peak load is one

        ax, ay = wave[:, None], wave[None, :]
        rho = self.aspect_ratio
        k2 = ax**2+(rho*ay)**2
        w_coeff = load_coeff/(k2**2)
        dbar, nu = 1.0, self.poisson
        return {
            "wave": wave,
            "load": load_coeff,
            "w": w_coeff,
            "beta_x": ax*w_coeff,
            "beta_y": ay*w_coeff,
            "M_xx": dbar*(ax**2+nu*(rho*ay)**2)*w_coeff,
            "M_yy": dbar*((rho*ay)**2+nu*ax**2)*w_coeff,
            "M_xy": -dbar*(1.0-nu)*rho*ax*ay*w_coeff,
            "Q_x": -dbar*ax*k2*w_coeff,
            "Q_y": -dbar*rho*ay*k2*w_coeff,
        }

    @property
    def output_scales(self) -> dict[str, np.ndarray]:
        """A-priori scales from the known load's first mode, not the reference solution."""
        q11, margin = abs(self.first_load_coefficient), self.output_scale_margin
        kx = ky = np.pi
        rho, nu = self.aspect_ratio, self.poisson
        k2 = kx**2+(rho*ky)**2
        w = q11/k2**2
        estimates = {
            "w": np.asarray([w]),
            "beta": np.asarray([kx*w, ky*w]),
            "moment": np.asarray([
                (kx**2+nu*(rho*ky)**2)*w,
                ((rho*ky)**2+nu*kx**2)*w,
                (1.0-nu)*rho*kx*ky*w,
            ]),
            "shear": np.asarray([kx*k2*w, rho*ky*k2*w]),
        }
        return {
            name: np.maximum(margin*np.abs(value), 1.0e-10)
            for name, value in estimates.items()
        }

    @property
    def residual_scales(self) -> dict[str, np.ndarray]:
        """Known-load RMS response scales, independent of reference labels."""
        response = {
            name: np.asarray(value, dtype=float)/(2.0*self.output_scale_margin)
            for name, value in self.output_scales.items()
        }
        return {
            "kinematic": response["beta"],
            "constitutive": response["moment"],
            "moment": response["shear"],
            "equilibrium": np.asarray([self.load_rms]),
        }

    @property
    def residual_scale_source(self) -> str:
        return "RMS scales from known-load first mode and prescribed Gaussian load"

    @property
    def output_scale_source(self) -> str:
        return "known-load first sine mode followed by physics-only pretraining calibration"

    def load_numpy(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        cx, cy = self.normalized_center
        rx, ry = self.normalized_radii
        return np.exp(-((xy[:, 0]-cx)/rx)**2-((xy[:, 1]-cy)/ry)**2)

    def load_tf(self, xy):
        import tensorflow as tf
        cx, cy = self.normalized_center
        rx, ry = self.normalized_radii
        return tf.exp(-((xy[:, 0]-cx)/rx)**2-((xy[:, 1]-cy)/ry)**2)

    def rigidity_tf(self, xy):
        import tensorflow as tf
        # Dbar=D/Dc=1 because the physical rigidity is the reference rigidity.
        return tf.ones_like(xy[:, 0])

    def constitutive_target_tf(self, beta_gradient, xy):
        import tensorflow as tf
        rho, nu = self.aspect_ratio, self.poisson
        exx = beta_gradient[:, 0, 0]
        eyy = rho**2*beta_gradient[:, 1, 1]
        exy = 0.5*rho*(beta_gradient[:, 0, 1]+beta_gradient[:, 1, 0])
        dbar = self.rigidity_tf(xy)
        return tf.stack([
            -dbar*(exx+nu*eyy),
            -dbar*(eyy+nu*exx),
            -dbar*(1.0-nu)*exy,
        ], axis=1)

    @staticmethod
    def _project(left: np.ndarray, coefficient: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.sum((left@coefficient)*right, axis=1)

    def exact_numpy(self, xy: np.ndarray) -> dict[str, np.ndarray]:
        """Converged Navier-series reference fields on the unit square."""
        xy = np.asarray(xy, dtype=float)
        modal, wave = self.modal_data, self.modal_data["wave"]
        result = {
            "w": np.empty(len(xy)),
            "beta": np.empty((len(xy), 2)),
            "moment": np.empty((len(xy), 3)),
            "shear": np.empty((len(xy), 2)),
        }
        chunk_size = 4096
        for start in range(0, len(xy), chunk_size):
            stop = min(start+chunk_size, len(xy))
            points = xy[start:stop]
            sx, cx = np.sin(np.outer(points[:, 0], wave)), np.cos(np.outer(points[:, 0], wave))
            sy, cy = np.sin(np.outer(points[:, 1], wave)), np.cos(np.outer(points[:, 1], wave))
            result["w"][start:stop] = self._project(sx, modal["w"], sy)
            result["beta"][start:stop, 0] = self._project(cx, modal["beta_x"], sy)
            result["beta"][start:stop, 1] = self._project(sx, modal["beta_y"], cy)
            result["moment"][start:stop, 0] = self._project(sx, modal["M_xx"], sy)
            result["moment"][start:stop, 1] = self._project(sx, modal["M_yy"], sy)
            result["moment"][start:stop, 2] = self._project(cx, modal["M_xy"], cy)
            result["shear"][start:stop, 0] = self._project(cx, modal["Q_x"], sy)
            result["shear"][start:stop, 1] = self._project(sx, modal["Q_y"], cy)
        result["load"] = self.load_numpy(xy)
        return result

    def series_load_numpy(self, xy: np.ndarray) -> np.ndarray:
        """Truncated load represented by the same reference sine basis."""
        xy = np.asarray(xy, dtype=float)
        wave = self.modal_data["wave"]
        sx, sy = np.sin(np.outer(xy[:, 0], wave)), np.sin(np.outer(xy[:, 1], wave))
        return self._project(sx, self.modal_data["load"], sy)


@dataclass(frozen=True)
class HeterogeneousRoofPlate:
    """Simply-supported heterogeneous rock roof on a unit computational square.

    The physical roof is isotropic at every point, while its scalar bending
    rigidity contains a smooth low-rigidity zone.  A sine Ritz discretization
    supplies an independent reference solution for the variable-coefficient
    Kirchhoff problem.
    """

    physical_lx: float = 6.0
    physical_ly: float = 3.0
    reference_rigidity: float = 1.5e10
    poisson: float = 0.25
    load_amplitude: float = 5.0e5
    weak_zone_center_x: float = 3.8
    weak_zone_center_y: float = 1.5
    weak_zone_radius_x: float = 0.75
    weak_zone_radius_y: float = 0.60
    weak_zone_reduction: float = 0.65
    reference_modes: int = 33
    reference_quadrature_order: int = 80
    output_scale_margin: float = 1.5

    def __post_init__(self):
        if self.physical_lx <= 0.0 or self.physical_ly <= 0.0:
            raise ValueError("physical roof lengths must be positive")
        if self.reference_rigidity <= 0.0 or self.load_amplitude <= 0.0:
            raise ValueError("reference rigidity and load amplitude must be positive")
        if not 0.0 <= self.weak_zone_reduction < 1.0:
            raise ValueError("weak_zone_reduction must lie in [0, 1)")
        if self.weak_zone_radius_x <= 0.0 or self.weak_zone_radius_y <= 0.0:
            raise ValueError("weak-zone radii must be positive")
        if not (0.0 <= self.weak_zone_center_x <= self.physical_lx):
            raise ValueError("weak-zone x coordinate lies outside the roof")
        if not (0.0 <= self.weak_zone_center_y <= self.physical_ly):
            raise ValueError("weak-zone y coordinate lies outside the roof")
        if self.reference_modes < 1 or self.reference_quadrature_order < 2:
            raise ValueError("reference Ritz orders are invalid")

    @property
    def lx(self) -> float:
        return 1.0

    @property
    def ly(self) -> float:
        return 1.0

    @property
    def aspect_ratio(self) -> float:
        return self.physical_lx/self.physical_ly

    @property
    def normalized_weak_zone(self) -> tuple[float, float, float, float]:
        return (
            self.weak_zone_center_x/self.physical_lx,
            self.weak_zone_center_y/self.physical_ly,
            self.weak_zone_radius_x/self.physical_lx,
            self.weak_zone_radius_y/self.physical_ly,
        )

    @property
    def divergence_factors(self) -> tuple[float, float]:
        return 1.0, self.aspect_ratio

    @property
    def reference_label(self) -> str:
        return "高阶Ritz参考解"

    @property
    def coordinate_labels(self) -> tuple[str, str]:
        return r"$\xi$", r"$\eta$"

    @property
    def load_plot_title(self) -> str:
        return "平滑分布围岩荷载"

    @property
    def coordinate_mapping(self) -> dict:
        return {
            "type": "dimensionless_unit_square",
            "physical_domain": [0.0, self.physical_lx, 0.0, self.physical_ly],
            "computational_domain": [0.0, 1.0, 0.0, 1.0],
            "mapping": "xi=x/Lx, eta=y/Ly",
            "aspect_ratio_Lx_over_Ly": self.aspect_ratio,
        }

    @property
    def physical_field_scales(self) -> dict[str, np.ndarray]:
        p, a, b, d = (
            self.load_amplitude, self.physical_lx,
            self.physical_ly, self.reference_rigidity,
        )
        wc = p*a**4/d
        return {
            "w": np.asarray([wc]),
            "beta": np.asarray([wc/a, wc/b]),
            "moment": np.asarray([p*a**2, p*a**2, p*a**2]),
            "shear": np.asarray([p*a, p*a]),
            "load": np.asarray([p]),
        }

    def dimensionalize_fields(self, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        scales = self.physical_field_scales
        result = {
            "w": np.asarray(fields["w"])*scales["w"][0],
            "beta": np.asarray(fields["beta"])*scales["beta"][None, :],
            "moment": np.asarray(fields["moment"])*scales["moment"][None, :],
            "shear": np.asarray(fields["shear"])*scales["shear"][None, :],
        }
        if "load" in fields:
            result["load"] = np.asarray(fields["load"])*scales["load"][0]
        return result

    def computational_to_physical(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return np.c_[xy[:, 0]*self.physical_lx, xy[:, 1]*self.physical_ly]

    @property
    def minimum_normalized_rigidity(self) -> float:
        return 1.0-self.weak_zone_reduction

    def rigidity_numpy(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        cx, cy, rx, ry = self.normalized_weak_zone
        gaussian = np.exp(-((xy[:, 0]-cx)/rx)**2-((xy[:, 1]-cy)/ry)**2)
        return 1.0-self.weak_zone_reduction*gaussian

    def rigidity_gradient_numpy(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        cx, cy, rx, ry = self.normalized_weak_zone
        gaussian = np.exp(-((xy[:, 0]-cx)/rx)**2-((xy[:, 1]-cy)/ry)**2)
        factor = self.weak_zone_reduction*gaussian
        return np.c_[
            2.0*factor*(xy[:, 0]-cx)/rx**2,
            2.0*factor*(xy[:, 1]-cy)/ry**2,
        ]

    def rigidity_tf(self, xy):
        import tensorflow as tf
        cx, cy, rx, ry = self.normalized_weak_zone
        gaussian = tf.exp(-((xy[:, 0]-cx)/rx)**2-((xy[:, 1]-cy)/ry)**2)
        return 1.0-self.weak_zone_reduction*gaussian

    def load_numpy(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return np.sin(np.pi*xy[:, 0])*np.sin(np.pi*xy[:, 1])

    def load_tf(self, xy):
        import tensorflow as tf
        return tf.sin(np.pi*xy[:, 0])*tf.sin(np.pi*xy[:, 1])

    def constitutive_target_tf(self, beta_gradient, xy):
        import tensorflow as tf
        rho, nu = self.aspect_ratio, self.poisson
        exx = beta_gradient[:, 0, 0]
        eyy = rho**2*beta_gradient[:, 1, 1]
        exy = 0.5*rho*(beta_gradient[:, 0, 1]+beta_gradient[:, 1, 0])
        dbar = self.rigidity_tf(xy)
        return tf.stack([
            -dbar*(exx+nu*eyy),
            -dbar*(eyy+nu*exx),
            -dbar*(1.0-nu)*exy,
        ], axis=1)

    @property
    def output_scales(self) -> dict[str, np.ndarray]:
        """Conservative a-priori scales using the known load and minimum rigidity."""
        q11 = 1.0
        kx = ky = np.pi
        rho, nu = self.aspect_ratio, self.poisson
        dbar = self.minimum_normalized_rigidity
        k2 = kx**2+(rho*ky)**2
        w = q11/(dbar*k2**2)
        estimates = {
            "w": np.asarray([w]),
            "beta": np.asarray([kx*w, ky*w]),
            "moment": dbar*np.asarray([
                (kx**2+nu*(rho*ky)**2)*w,
                ((rho*ky)**2+nu*kx**2)*w,
                (1.0-nu)*rho*kx*ky*w,
            ]),
            "shear": dbar*np.asarray([kx*k2*w, rho*ky*k2*w]),
        }
        return {
            name: np.maximum(self.output_scale_margin*np.abs(value), 1.0e-10)
            for name, value in estimates.items()
        }

    @property
    def residual_scales(self) -> dict[str, np.ndarray]:
        """Conservative RMS scales from known loading and minimum rigidity."""
        response = {
            name: np.asarray(value, dtype=float)/(2.0*self.output_scale_margin)
            for name, value in self.output_scales.items()
        }
        return {
            "kinematic": response["beta"],
            "constitutive": response["moment"],
            "moment": response["shear"],
            "equilibrium": np.asarray([0.5]),
        }

    @property
    def residual_scale_source(self) -> str:
        return "RMS scales from prescribed load, wave numbers and minimum rigidity"

    @property
    def output_scale_source(self) -> str:
        return "prescribed-load first sine mode and prescribed minimum rigidity"

    @staticmethod
    def _project(left: np.ndarray, coefficient: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.sum((left@coefficient)*right, axis=1)

    @cached_property
    def ritz_data(self) -> dict[str, np.ndarray]:
        """Assemble and solve the variable-rigidity sine Ritz reference system."""
        from numpy.polynomial.legendre import leggauss

        order, count = self.reference_quadrature_order, self.reference_modes
        z, wz = leggauss(order)
        points, weights = 0.5*(z+1.0), 0.5*wz
        wave = np.pi*np.arange(1, count+1, dtype=float)
        sine, cosine = np.sin(np.outer(points, wave)), np.cos(np.outer(points, wave))

        phi = np.einsum("im,jn->ijmn", sine, sine).reshape(order**2, count**2)
        kxx = np.einsum("im,jn,m->ijmn", sine, sine, -wave**2).reshape(order**2, count**2)
        kyy = np.einsum("im,jn,n->ijmn", sine, sine, -wave**2).reshape(order**2, count**2)
        kxy = np.einsum("im,jn,m,n->ijmn", cosine, cosine, wave, wave).reshape(order**2, count**2)

        xx, yy = np.meshgrid(points, points, indexing="ij")
        xy = np.c_[xx.ravel(), yy.ravel()]
        integration_weights = (weights[:, None]*weights[None, :]).ravel()
        weighted_rigidity = integration_weights*self.rigidity_numpy(xy)
        rho, nu = self.aspect_ratio, self.poisson
        ky_phys = rho**2*kyy
        kxy_phys = rho*kxy

        stiffness = (
            kxx.T@(weighted_rigidity[:, None]*kxx)
            + ky_phys.T@(weighted_rigidity[:, None]*ky_phys)
            + nu*kxx.T@(weighted_rigidity[:, None]*ky_phys)
            + nu*ky_phys.T@(weighted_rigidity[:, None]*kxx)
            + 2.0*(1.0-nu)*kxy_phys.T@(weighted_rigidity[:, None]*kxy_phys)
        )
        force = phi.T@(integration_weights*self.load_numpy(xy))
        coefficients = np.linalg.solve(stiffness, force).reshape(count, count)
        residual = stiffness@coefficients.ravel()-force
        return {
            "wave": wave,
            "coefficients": coefficients,
            "relative_algebraic_residual": np.linalg.norm(residual)/(np.linalg.norm(force)+1.0e-30),
        }

    def exact_numpy(self, xy: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate the independent variable-rigidity Ritz reference fields."""
        xy = np.asarray(xy, dtype=float)
        wave = self.ritz_data["wave"]
        coefficients = self.ritz_data["coefficients"]
        rho, nu = self.aspect_ratio, self.poisson
        result = {
            "w": np.empty(len(xy)),
            "beta": np.empty((len(xy), 2)),
            "moment": np.empty((len(xy), 3)),
            "shear": np.empty((len(xy), 2)),
        }
        wx = wave[:, None]
        wy = wave[None, :]
        chunk_size = 4096
        for start in range(0, len(xy), chunk_size):
            stop = min(start+chunk_size, len(xy))
            points = xy[start:stop]
            sx = np.sin(np.outer(points[:, 0], wave))
            cx = np.cos(np.outer(points[:, 0], wave))
            sy = np.sin(np.outer(points[:, 1], wave))
            cy = np.cos(np.outer(points[:, 1], wave))

            w = self._project(sx, coefficients, sy)
            beta_x = self._project(cx, wx*coefficients, sy)
            beta_y = self._project(sx, coefficients*wy, cy)
            wxx = self._project(sx, -(wx**2)*coefficients, sy)
            wyy = self._project(sx, -(wy**2)*coefficients, sy)
            wxy = self._project(cx, wx*coefficients*wy, cy)
            wxxx = self._project(cx, -(wx**3)*coefficients, sy)
            wxyy = self._project(cx, -wx*coefficients*(wy**2), sy)
            wxxy = self._project(sx, -(wx**2)*coefficients*wy, cy)
            wyyy = self._project(sx, -(wy**3)*coefficients, cy)

            dbar = self.rigidity_numpy(points)
            grad_d = self.rigidity_gradient_numpy(points)
            a = wxx+nu*rho**2*wyy
            b = rho**2*wyy+nu*wxx
            c = (1.0-nu)*rho*wxy
            ax = wxxx+nu*rho**2*wxyy
            by = rho**2*wyyy+nu*wxxy
            cx_term = (1.0-nu)*rho*wxxy
            cy_term = (1.0-nu)*rho*wxyy

            result["w"][start:stop] = w
            result["beta"][start:stop] = np.c_[beta_x, beta_y]
            result["moment"][start:stop] = -dbar[:, None]*np.c_[a, b, c]
            result["shear"][start:stop, 0] = (
                grad_d[:, 0]*a+dbar*ax+rho*(grad_d[:, 1]*c+dbar*cy_term)
            )
            result["shear"][start:stop, 1] = (
                grad_d[:, 0]*c+dbar*cx_term+rho*(grad_d[:, 1]*b+dbar*by)
            )
        result["load"] = self.load_numpy(xy)
        return result


def build_problem(config) -> NavierPlate | LocalGaussianPlate | HeterogeneousRoofPlate:
    """Build one of the three verification problems from a resolved config."""
    dc, pc = config["domain"], config["physics"]
    kind = pc.get("problem", "navier_sine")
    if kind == "navier_sine":
        return NavierPlate(
            dc["lx"], dc["ly"], pc["D"], pc["nu"], pc["load_amplitude"],
            pc.get("output_scale_margin", 2.0),
        )
    if kind == "local_gaussian":
        if not (np.isclose(dc["lx"], 1.0) and np.isclose(dc["ly"], 1.0)):
            raise ValueError("local_gaussian uses the computational unit square: domain.lx=domain.ly=1")
        center = pc.get("load_center", [dc.get("physical_lx", 2.0)/2.0, dc.get("physical_ly", 1.0)/2.0])
        radius = pc.get("load_radius", [0.16, 0.16])
        return LocalGaussianPlate(
            physical_lx=dc.get("physical_lx", 2.0),
            physical_ly=dc.get("physical_ly", 1.0),
            rigidity=pc["D"],
            poisson=pc["nu"],
            load_amplitude=pc["load_amplitude"],
            load_center_x=center[0],
            load_center_y=center[1],
            load_radius_x=radius[0],
            load_radius_y=radius[1],
            reference_terms=pc.get("reference_terms", 61),
            coefficient_quadrature_order=pc.get("coefficient_quadrature_order", 256),
            output_scale_margin=pc.get("output_scale_margin", 1.5),
        )
    if kind == "heterogeneous_roof":
        if not (np.isclose(dc["lx"], 1.0) and np.isclose(dc["ly"], 1.0)):
            raise ValueError("heterogeneous_roof uses the computational unit square: domain.lx=domain.ly=1")
        weak_center = pc.get("weak_zone_center", [3.8, 1.5])
        weak_radius = pc.get("weak_zone_radius", [0.75, 0.60])
        return HeterogeneousRoofPlate(
            physical_lx=dc.get("physical_lx", 6.0),
            physical_ly=dc.get("physical_ly", 3.0),
            reference_rigidity=pc["D"],
            poisson=pc["nu"],
            load_amplitude=pc["load_amplitude"],
            weak_zone_center_x=weak_center[0],
            weak_zone_center_y=weak_center[1],
            weak_zone_radius_x=weak_radius[0],
            weak_zone_radius_y=weak_radius[1],
            weak_zone_reduction=pc.get("weak_zone_reduction", 0.65),
            reference_modes=pc.get("reference_modes", 33),
            reference_quadrature_order=pc.get("reference_quadrature_order", 80),
            output_scale_margin=pc.get("output_scale_margin", 1.5),
        )
    raise ValueError(f"Unknown physics.problem: {kind!r}")
