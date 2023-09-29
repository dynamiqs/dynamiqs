from __future__ import annotations

from abc import abstractmethod

import numpy as np
from torch import Tensor

from .solver import AutogradSolver
from .utils import merge_tensors
from .utils.td_tensor import CallableTDTensor, PWCTDTensor
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

        if isinstance(self.H, CallableTDTensor):
            raise ValueError(
                'Solver `Propagator` requires a time-independent or piecewise-constant'
                ' Hamiltonian, but a time-dependent Hamiltonian in the callable format'
                ' was provided.'
            )
        elif isinstance(self.H, PWCTDTensor):
            # merge tstop with pwc times that are in the interval [0.0, tsave[-1]],
            # to make sure no piecewise constant part is skipped during the evolution
            mask = (0.0 <= self.H.times) & (self.H.times <= self.tsave[-1])
            additional_tstop = self.H.times[mask]
            self.tstop = merge_tensors(self.tstop, additional_tstop)

    def run_autograd(self):
        y, t1 = self.y0, 0.0
        for t2 in tqdm(self.tstop.cpu().numpy(), disable=not self.options.verbose):
            if t2 != 0.0:
                # round time difference to avoid numerical errors when comparing floats
                delta_t = round_truncate(t2 - t1)
                y = self.forward(t1, delta_t, y)
            if t2 in self.tsave:
                self.save(y)
            t1 = t2

    @abstractmethod
    def forward(self, t: float, delta_t: float, y: Tensor) -> Tensor:
        pass
