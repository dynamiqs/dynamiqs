import jax
from jaxtyping import Scalar

from ..core.abstract_integrator import SESolveIntegrator
from ..core.propagator_integrator import PropagatorIntegrator
from ..qarrays import QArray


class SESolvePropagatorIntegrator(PropagatorIntegrator, SESolveIntegrator):
    # supports only ConstantTimeArray
    # TODO: support PWCTimeArray

    def forward(self, delta_t: Scalar, y: QArray) -> QArray:
        propagator = jax.scipy.linalg.expm(-1j * self.H * delta_t)
        return propagator @ y
