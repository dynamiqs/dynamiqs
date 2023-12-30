from jax import Array

from ..solvers.abstract import FixedStepSolver
from ..solvers.base import MEIterativeSolver
from ..types import Scalar
from ..utils.utils import dag


class MEEuler(MEIterativeSolver, FixedStepSolver):
    @property
    def dt(self) -> Scalar:
        return self.options.dt

    def forward(self, t: Scalar, y: Array) -> Array:
        H = self.H(t)
        tmp = -1j * H @ y
        return y + self.dt * (tmp + dag(tmp))
