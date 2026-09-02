from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss, Legendre


@dataclass(frozen=True)
class RectMesh:
    lx: float
    ly: float
    nx: int
    ny: int

    @property
    def cells(self) -> np.ndarray:
        xs, ys = np.linspace(0.0, self.lx, self.nx+1), np.linspace(0.0, self.ly, self.ny+1)
        return np.asarray([(xs[i], xs[i+1], ys[j], ys[j+1]) for j in range(self.ny) for i in range(self.nx)])


def legendre_bubbles(order: int, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bubble basis b_k=(1-xi^2)P_k and derivatives on [-1,1]."""
    vals, ders = [], []
    for k in range(order):
        p = Legendre.basis(k)
        vals.append((1.0-xi**2)*p(xi))
        ders.append(-2.0*xi*p(xi)+(1.0-xi**2)*p.deriv()(xi))
    return np.asarray(vals), np.asarray(ders)


def cell_rule(cell: np.ndarray, quadrature_order: int, test_order: int) -> dict[str, np.ndarray]:
    x0, x1, y0, y1 = map(float, cell)
    z, wz = leggauss(quadrature_order)
    xx, yy = np.meshgrid(z, z, indexing="ij")
    xy = np.c_[x0+(xx.ravel()+1)*(x1-x0)/2, y0+(yy.ravel()+1)*(y1-y0)/2]
    weights = np.outer(wz, wz).ravel()*(x1-x0)*(y1-y0)/4
    bx, dbx = legendre_bubbles(test_order, z)
    by, dby = legendre_bubbles(test_order, z)
    phi, gx, gy = [], [], []
    for i in range(test_order):
        for j in range(test_order):
            phi.append(np.outer(bx[i], by[j]).ravel())
            gx.append(np.outer(dbx[i]*2/(x1-x0), by[j]).ravel())
            gy.append(np.outer(bx[i], dby[j]*2/(y1-y0)).ravel())
    return {"xy": xy, "weights": weights, "phi": np.asarray(phi), "grad_phi": np.stack([gx, gy], axis=-1), "area": (x1-x0)*(y1-y0)}


def batched_cell_rule(cells: np.ndarray, quadrature_order: int, test_order: int) -> dict[str, np.ndarray]:
    """Stack equal-order cell rules for vectorized weak-form evaluation."""
    rules = [cell_rule(cell, quadrature_order, test_order) for cell in np.asarray(cells)]
    if not rules:
        raise ValueError("At least one cell is required")
    return {
        "xy": np.stack([r["xy"] for r in rules]),
        "weights": np.stack([r["weights"] for r in rules]),
        "phi": np.stack([r["phi"] for r in rules]),
        "grad_phi": np.stack([r["grad_phi"] for r in rules]),
        "area": np.asarray([r["area"] for r in rules]),
    }
