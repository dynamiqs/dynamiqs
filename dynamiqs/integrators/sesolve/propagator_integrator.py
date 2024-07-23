import jax
from jax import Array
from jaxtyping import Scalar

from ..core.propagator_solver import _SESolvePropagatorIntegrator


class SESolvePropagatorIntegrator(_SESolvePropagatorIntegrator):
    # supports only ConstantTimeArray
    # TODO: support PWCTimeArray

    def forward(self, delta_t: Scalar, y: Array) -> Array:
        propagator = jax.scipy.linalg.expm(-1j * self.H * delta_t)
        return propagator @ y
