from __future__ import annotations

from abc import abstractmethod

import numpy as np
from torch import Tensor

from .solver import AutogradSolver
from .utils.td_tensor import ConstantTDTensor
from .utils.utils import tqdm


def round_truncate(x: np.float32 | np.float64) -> np.float32 | np.float64:
    # round a strictly positive-valued float to remove numerical errors, and enable
    # comparing floats for equality

    # The mantissa of a float32 is stored using 23 bits. The following code rounds and
    # truncates the float value to the 18 most significant bits of its mantissa. This
    # removes any numerical error that may have accumulated in the 5 least significant
    # bits of the mantissa.
    leading = abs(int(np.log2(x)))
    kept = leading + 18
    return (x * 2**kept).round() / 2**kept


class Propagator(AutogradSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check that Hamiltonian is time-independent
        if not isinstance(self.H, ConstantTDTensor):
            raise TypeError(
                'Solver `Propagator` requires a time-independent Hamiltonian.'
            )
        self.H = self.H(0.0)

    def run_autograd(self):
        y, t1 = self.y0, 0.0
        for t2 in tqdm(self.t_stop.cpu().numpy(), disable=not self.options.verbose):
            if t2 != 0.0:
                # round time difference to avoid numerical errors when comparing floats
                delta_t = round_truncate(t2 - t1)
                y = self.forward(t1, delta_t, y)
            self.save(y)
            t1 = t2

    @abstractmethod
    def forward(self, t: float, delta_t: float, y: Tensor):
        pass
