from typing import Callable

import diffrax as dx
from jaxtyping import PyTree, Scalar

from ..time_array import TimeArray


class SchrodingerTerm(dx.ODETerm):
    H: TimeArray  # (n, n)
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree]

    def __init__(self, H: TimeArray):
        self.H = H

    def vector_field(self, t: Scalar, psi: PyTree, _args: PyTree):
        return -1j * self.H(t) @ psi
