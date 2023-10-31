from __future__ import annotations

from abc import ABC, abstractmethod
from math import inf
from time import time

import torch
from torch import Tensor

from .options import Options
from .result import Result
from .utils.td_tensor import TDTensor
from .utils.utils import bexpect, iteraxis


class Solver(ABC):
    def __init__(
        self,
        H: TDTensor,
        y0: Tensor,
        tsave: Tensor,
        tmeas: Tensor,
        exp_ops: Tensor,
        options: Options,
    ):
        """

        Args:
            H:
            y0: Initial quantum state, of shape `(..., m, n)`.
            tsave: Times for which results are saved.
            exp_ops:
            options:
        """
        self.H = H
        self.y0 = y0
        self.tsave = tsave
        self.tmeas = tmeas
        self.exp_ops = exp_ops
        self.options = options

        # aliases
        self.cdtype = self.options.cdtype
        self.rdtype = self.options.rdtype
        self.device = self.options.device

        # initialize time logic
        self.tstop = torch.cat((self.tsave, self.tmeas)).unique().sort()[0]
        self.tstop_counter = 0
        self.tsave_counter = 0
        self.tmeas_counter = 0

        # initialize save tensors
        batch_sizes, (m, n) = y0.shape[:-2], y0.shape[-2:]

        if self.options.save_states:
            # ysave: (..., len(tsave), m, n)
            self.ysave = torch.zeros(
                *batch_sizes, len(tsave), m, n, dtype=self.cdtype, device=self.device
            )
            self.ysave_iter = iteraxis(self.ysave, axis=-3)
        else:
            self.ysave = None

        if len(self.exp_ops) > 0:
            # exp_save: (..., len(exp_ops), len(tsave))
            self.exp_save = torch.zeros(
                *batch_sizes,
                len(exp_ops),
                len(tsave),
                dtype=self.cdtype,
                device=self.device,
            )
            self.exp_save_iter = iteraxis(self.exp_save, axis=-1)
        else:
            self.exp_save = None

    def run(self) -> Result:
        start_time = time()
        self._run()
        end_time = time()

        result = Result(self.options, self.ysave, self.tsave, self.exp_save)
        result.start_time = start_time
        result.end_time = end_time
        return result

    def _run(self):
        if self.options.gradient_alg is None:
            self.run_nograd()

    @abstractmethod
    def run_nograd(self):
        pass

    def next_tsave(self) -> float:
        if self.tsave_counter == len(self.tsave):
            return inf
        return self.tsave[self.tsave_counter].item()

    def next_tmeas(self) -> float:
        if self.tmeas_counter == len(self.tmeas):
            return inf
        return self.tmeas[self.tmeas_counter].item()

    def next_tstop(self) -> float:
        return self.tstop[self.tstop_counter].item()

    def save(self, y: Tensor):
        if self.next_tstop() == self.next_tsave():
            self._save_y(y)
            self._save_exp_ops(y)
            self.tsave_counter += 1
        if self.next_tstop() == self.next_tmeas():
            self._save_meas()
            self.tmeas_counter += 1
        self.tstop_counter += 1

    def _save_y(self, y: Tensor):
        if self.options.save_states:
            next(self.ysave_iter)[:] = y
        # otherwise only save the state if it is the final state
        elif self.next_tsave() is inf:
            self.ysave = y

    def _save_exp_ops(self, y: Tensor):
        if len(self.exp_ops) > 0:
            next(self.exp_save_iter)[:] = bexpect(self.exp_ops, y)

    def _save_meas(self):
        pass


class AutogradSolver(Solver):
    def _run(self):
        super()._run()
        if self.options.gradient_alg == 'autograd':
            self.run_autograd()

    def run_nograd(self):
        with torch.inference_mode():
            self.run_autograd()

    @abstractmethod
    def run_autograd(self):
        pass


class AdjointSolver(AutogradSolver):
    def _run(self):
        super()._run()
        if self.options.gradient_alg == 'adjoint':
            self.run_adjoint()

    @abstractmethod
    def run_adjoint(self):
        """Integrate an ODE using the adjoint method in the backward pass."""
        pass
