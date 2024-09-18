from jaxtyping import PyTree

from ...result import Saved
from ...utils.vectorization import operator_to_vector, vector_to_operator
from ..core.abstract_integrator import MESolveIntegrator
from ..core.expm_integrator import MEExpmIntegrator


class MESolveExpmIntegrator(MEExpmIntegrator, MESolveIntegrator):
    def __init__(self, *args):
        super().__init__(*args)

        # convert to vectorized form
        self.y0 = operator_to_vector(self.y0)  # (n^2, 1)

    def save(self, y: PyTree) -> Saved:
        # TODO: implement bexpect for vectorized operators and convert at the end
        #       instead ofat each step
        y = vector_to_operator(y)
        return super().save(y)
