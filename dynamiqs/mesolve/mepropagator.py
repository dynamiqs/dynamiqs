import jax
from jax import Array
from jaxtyping import Scalar

from ..core.propagator_solver import MEPropagatorSolver
from ..result import Saved
from ..utils.vectorization import operator_to_vector, slindbladian, vector_to_operator


class MEPropagator(MEPropagatorSolver):
    # supports only ConstantTimeArray
    # TODO: support PWCTimeArray
    lindbladian: Array

    def __init__(self, *args):
        super().__init__(*args)
        self.lindbladian = slindbladian(self.H, self.Ls)  # (n^2, n^2)
        self.y0 = operator_to_vector(self.y0)  # (n^2, 1)

    def forward(self, delta_t: Scalar, y: Array) -> Array:
        propagator = jax.scipy.linalg.expm(delta_t * self.lindbladian)
        return propagator @ y

    def save(self, y: Array) -> Saved:
        # TODO: implement bexpect for vectorized operators and convert at the end
        #       instead ofat each step
        y = vector_to_operator(y)
        return super().save(y)
