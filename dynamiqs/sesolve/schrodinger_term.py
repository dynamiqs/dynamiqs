from __future__ import annotations

from typing import Callable, Optional

from jaxtyping import PyTree, Scalar

from progress_bar import ProgressBarTerm
from ..time_array import TimeArray


class SchrodingerTerm(ProgressBarTerm):
    H: TimeArray  # (n, n)
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree]

    def __init__(self, H: TimeArray, update_progressbar: Optional[Callable]):
        super().__init__(update_progressbar)
        self.H = H

    def vector_field(self, t: Scalar, psi: PyTree, _args: PyTree):
        super().vector_field(t, psi, _args)
        return -1j * self.H(t) @ psi
