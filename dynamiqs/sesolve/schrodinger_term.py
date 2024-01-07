from typing import Callable

import diffrax as dx
from jaxtyping import PyTree, Scalar

from .._utils import merge_complex, split_complex
from ..time_array import ConstantTimeArray, TimeArray


class SchrodingerTerm(dx.ODETerm):
    H: TimeArray  # (n, n)
    constant_H: bool
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree]

    def __init__(self, H: TimeArray):
        self.constant_H = isinstance(H, ConstantTimeArray)
        if self.constant_H:
            self.H = H(0.0)
        else:
            self.H = H

    def vector_field(self, t: Scalar, psi: PyTree, _args: PyTree):
        if self.constant_H:
            H = self.H
        else:
            H = self.H(t)
        psi = merge_complex(psi)
        res = -1j * H @ psi
        return split_complex(res)
