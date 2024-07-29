import jax
from jaxtyping import Scalar

from ...qarrays import QArray
from ..core.abstract_integrator import SESolveIntegrator
from ..core.propagator_integrator import PropagatorIntegrator


class SESolvePropagatorIntegrator(PropagatorIntegrator, SESolveIntegrator):
    # supports only ConstantTimeArray
    # TODO: support PWCTimeArray

    def forward(self, delta_t: Scalar, y: QArray) -> QArray:
        propagator = jax.scipy.linalg.expm(-1j * self.H * delta_t)
        return propagator @ y
