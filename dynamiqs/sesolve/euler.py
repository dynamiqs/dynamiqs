from jax import Array

from ..solvers.abstract import FixedStepSolver
from ..solvers.base import SEIterativeSolver
from ..types import Scalar


class SEEuler(SEIterativeSolver, FixedStepSolver):
    @property
    def dt(self) -> Scalar:
        return self.options.dt

    def forward(self, t: Scalar, y: Array) -> Array:
        H = self.H(t)
        return y - 1j * self.dt * H @ y
