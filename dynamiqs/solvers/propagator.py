from __future__ import annotations

from abc import abstractmethod

import numpy as np
from torch import Tensor

from .._utils import obj_type_str
from ..time_tensor import ConstantTimeTensor, PWCTimeTensor
from .solver import AutogradSolver
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
        if not isinstance(self.H, (ConstantTimeTensor, PWCTimeTensor)):
            raise TypeError(
                f'Solver {obj_type_str(self.options.solver)} requires a'
                ' time-independent or piecewise-constant Hamiltonian (types'
                ' `ArrayLike`, `ConstantTimeTensor` or `PWCTimeTensor`), but the'
                f' passed Hamiltonian has type {obj_type_str(self.H)}.'
            )
        if isinstance(self.H, PWCTimeTensor):
            # We build a custom times array `tstop_new` gathering `self.tstop` and
            # `self.H.times`. This ensures that each PWC part of the Hamiltonian is
            # taken into account during the evolution.

            # keep only `H.times` values in the interval [t0, tsave[-1])
            mask = (self.t0 <= self.H.times) & (self.H.times < self.tsave[-1])
            extra_tstop = self.H.times[mask]
            extra_tstop = extra_tstop.numpy(force=True)

            # build new times to stop
            self.tstop_new = np.unique(np.concatenate((self.tstop, extra_tstop)))
        else:
            self.tstop_new = self.tstop

    def run_autograd(self):
        t1, y = self.t0, self.y0
        for t2 in tqdm(self.tstop_new, disable=not self.options.verbose):
            if t2 != self.t0:
                # round time difference to avoid numerical errors when comparing floats
                delta_t = round_truncate(t2 - t1)
                y = self.forward(t1, delta_t, y)
            # if the time is in the original `self.tstop` times, save the result
            if t2 in self.tstop:
                self.save(y)
            t1 = t2

    @abstractmethod
    def forward(self, t: float, delta_t: float, y: Tensor) -> Tensor:
        pass
