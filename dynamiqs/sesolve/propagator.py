import jax
from jax import Array

from ..solvers.abstract import PropagatorSolver
from ..solvers.base import SEIterativeSolver
from ..types import Scalar


class SEPropagator(SEIterativeSolver, PropagatorSolver):
    def __init__(self, *args):
        super().__init__(*args)
        self.H = self.H(0.0)

    def forward(self, t: Scalar, delta_t: Scalar, y: Array) -> Array:
        propagator = jax.scipy.linalg.expm(-1j * self.H * delta_t)
        return propagator @ y
