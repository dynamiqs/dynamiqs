import jax
from jaxtyping import Scalar

from ..core.propagator_solver import SEPropagatorSolver
from ..qarrays import QArray


class SEPropagator(SEPropagatorSolver):
    # supports only ConstantTimeArray
    # TODO: support PWCTimeArray

    def forward(self, delta_t: Scalar, y: QArray) -> QArray:
        propagator = jax.scipy.linalg.expm(-1j * self.H * delta_t)
        return propagator @ y
