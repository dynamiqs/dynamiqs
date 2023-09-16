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
    keep = leading + 18
    return (x * 2**keep).round() / 2**keep


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
        # initialize time and state
        y, t = self.y0, 0.0

        # save initial state
        if self.t_save[0] == 0.0:
            self.save(y)

        # run the ode routine
        nobar = not self.options.verbose
        for ts in tqdm(self.t_stop(), disable=nobar):
            # round time difference to avoid numerical errors when comparing floats
            delta_t = round_truncate(ts - t)
            y = self.forward(t, delta_t, y)

            # save state
            self.save(y)

            # iterate time
            t = ts

    @abstractmethod
    def forward(self, t: float, delta_t: float, y: Tensor):
        pass
