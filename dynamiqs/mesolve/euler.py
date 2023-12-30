from jax import Array

from ..solvers.abstract import FixedStepSolver
from ..solvers.base import MEIterativeSolver
from ..types import Scalar


class MEEuler(MEIterativeSolver, FixedStepSolver):
    @property
    def dt(self) -> Scalar:
        return self.options.dt

    def forward(self, t: Scalar, y: Array) -> Array:
        dy = self.dt * self.lindbladian(t, y)
        return y + dy
