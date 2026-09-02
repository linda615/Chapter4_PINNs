"""Mixed local weak-form PCRB-Net."""

from .problem import LocalGaussianPlate, NavierPlate, build_problem

__all__ = ["NavierPlate", "LocalGaussianPlate", "build_problem"]
