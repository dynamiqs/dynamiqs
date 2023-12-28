from __future__ import annotations

from abc import ABC, abstractmethod
from time import time

import numpy as np
import torch
from torch import Tensor

from ..gradient import Adjoint, Autograd
from ..time_tensor import TimeTensor
from .options import Options
from .result import Result
from .utils.utils import bexpect, iteraxis


class Solver(ABC):
    def __init__(
        self,
        H: TimeTensor,
        y0: Tensor,
        tsave: Tensor,
        tmeas: Tensor,
        E: Tensor,
        options: Options,
    ):
        """

        Args:
            H:
            y0: Initial quantum state, of shape `(..., m, n)`.
            tsave: Times for which results are saved.
            E:
            options:
        """
        self.H = H
        self.t0 = 0.0
        self.y0 = y0
        self.tsave = tsave.numpy()
        self.tmeas = tmeas.numpy()
        self.E = E
        self.options = options

        # aliases
        self.cdtype = self.options.cdtype
        self.rdtype = self.options.rdtype
        self.device = self.options.device

        # initialize time logic
        self.tstop = np.unique(np.concatenate((self.tsave, self.tmeas)))
        self.tsave_mask = np.isin(self.tstop, self.tsave)
        self.tmeas_mask = np.isin(self.tstop, self.tmeas)
        self.tstop_counter = 0
        self.tsave_counter = 0
        self.tmeas_counter = 0

        # initialize save tensors
        batch_sizes, (m, n) = y0.shape[:-2], y0.shape[-2:]
        kw = dict(dtype=self.cdtype, device=self.device)

        if self.options.save_states:
            # ysave: (..., len(tsave), m, n)
            self.ysave = torch.zeros(*batch_sizes, len(tsave), m, n, **kw)
            self.ysave_iter = iteraxis(self.ysave, axis=-3)
        else:
            self.ysave = None

        if len(self.E) > 0:
            # Esave: (..., len(E), len(tsave))
            self.Esave = torch.zeros(*batch_sizes, len(E), len(tsave), **kw)
            self.Esave_iter = iteraxis(self.Esave, axis=-1)
        else:
            self.Esave = None

    def run(self) -> Result:
        start_time = time()
        self._run()
        end_time = time()

        result = Result(self.options, self.ysave, self.tsave, self.Esave)
        result.start_time = start_time
        result.end_time = end_time
        return result

    def _run(self):
        if self.options.gradient is None:
            self.run_nograd()

    @abstractmethod
    def run_nograd(self):
        pass

    def next_tstop(self) -> float:
        return self.tstop[self.tstop_counter]

    def save(self, y: Tensor):
        if self.tsave_mask[self.tstop_counter]:
            self._save_y(y)
            self._save_E(y)
            self.tsave_counter += 1
        if self.tmeas_mask[self.tstop_counter]:
            self._save_meas()
            self.tmeas_counter += 1
        self.tstop_counter += 1

    def _save_y(self, y: Tensor):
        if self.options.save_states:
            next(self.ysave_iter)[:] = y
        # otherwise only save the state if it is the final state
        elif self.tsave_counter == len(self.tsave) - 1:
            self.ysave = y

    def _save_E(self, y: Tensor):
        if len(self.E) > 0:
            next(self.Esave_iter)[:] = bexpect(self.E, y)

    def _save_meas(self):
        pass


class AutogradSolver(Solver):
    def _run(self):
        super()._run()
        if isinstance(self.options.gradient, Autograd):
            self.run_autograd()

    def run_nograd(self):
        with torch.no_grad():
            self.run_autograd()

    @abstractmethod
    def run_autograd(self):
        pass


class AdjointSolver(AutogradSolver):
    def _run(self):
        super()._run()
        if isinstance(self.options.gradient, Adjoint):
            self.run_adjoint()

    @abstractmethod
    def run_adjoint(self):
        """Integrate an ODE using the adjoint method in the backward pass."""
        pass
