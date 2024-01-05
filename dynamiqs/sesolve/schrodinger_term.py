import diffrax as dx
from jax.typing import PyTree, Scalar

from ..time_array import TimeArray


class SchrodingerTerm(dx.ODETerm):
    H: TimeArray  # (n, n)

    def vector_field(self, t: Scalar, psi: PyTree, _args: PyTree):
        return -1j * self.H(t) @ psi
