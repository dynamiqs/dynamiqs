from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

import torch
from torch import Tensor

from ..options import Options
from ..utils.solver_utils import bexpect
from ..utils.td_tensor import TDTensor
from ..utils.tensor_types import dtype_complex_to_real


class BaseSolver(ABC):
    def __init__(self, t_save_all: Tensor):
        self.t_save_all = t_save_all
        self.t_save_all_counter = 0

    def next_t_save_all(self) -> float:
        return self.t_save_all[self.t_save_all_counter].item()

    def save(self, y: Tensor):
        self._save(y)
        self.t_save_all_counter += 1

    @abstractmethod
    def _save(self, y: Tensor):
        pass


class Solver(BaseSolver):
    def __init__(
        self,
        H: TDTensor,
        y0: Tensor,
        t_save_all: Tensor,
        t_save: Tensor,
        exp_ops: Tensor,
        options: Options,
    ):
        """

        Args:
            H:
            y0: Initial quantum state, of shape `(..., m, n)`.
            t_save: Times for which results are saved.
            exp_ops:
            options:
        """
        super().__init__(t_save_all)
        self.H = H
        self.y0 = y0
        self.exp_ops = exp_ops
        self.options = options

        # save logic for states and exp_ops
        self.t_save_mask = torch.isin(self.t_save_all, t_save)
        self.t_save_counter = 0

        # initialize save tensors
        batch_sizes, (m, n) = self.y0.shape[:-2], self.y0.shape[-2:]

        if self.options.save_states:
            # y_save: (..., len(t_save), m, n)
            self.y_save = torch.zeros(
                *batch_sizes,
                len(t_save),
                m,
                n,
                dtype=self.dtype,
                device=self.device,
            )

        if len(self.exp_ops) > 0:
            # exp_save: (..., len(exp_ops), len(t_save))
            self.exp_save = torch.zeros(
                *batch_sizes,
                len(self.exp_ops),
                len(t_save),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.exp_save = None

    @cached_property
    def dtype(self) -> torch.complex64 | torch.complex128:
        return self.y0.dtype

    @cached_property
    def device(self) -> torch.device:
        return self.y0.device

    @cached_property
    def dtype_real(self) -> torch.float32 | torch.float64:
        return dtype_complex_to_real(self.y0.dtype)

    def run(self):
        if self.options.gradient_alg is None:
            self.run_nograd()

    @abstractmethod
    def run_nograd(self):
        pass

    def _save(self, y: Tensor):
        if self.t_save_mask[self.t_save_all_counter]:
            self._save_y(y)
            self._save_exp_ops(y)
            self.t_save_counter += 1

    def _save_y(self, y: Tensor):
        if self.options.save_states:
            self.y_save[..., self.t_save_counter, :, :] = y
        # otherwise only save the state if it is the final state
        elif self.t_save_counter == len(self.t_save) - 1:
            self.y_save = y

    def _save_exp_ops(self, y: Tensor):
        if len(self.exp_ops) > 0:
            self.exp_save[..., self.t_save_counter] = bexpect(self.exp_ops, y)


class AutogradSolver(Solver):
    def run(self):
        super().run()
        if self.options.gradient_alg == 'autograd':
            self.run_autograd()

    def run_nograd(self):
        with torch.no_grad():
            self.run_autograd()

    @abstractmethod
    def run_autograd(self):
        pass


class AdjointSolver(AutogradSolver):
    def run(self):
        super().run()
        if self.options.gradient_alg == 'adjoint':
            self.run_adjoint()

    @abstractmethod
    def run_adjoint(self):
        """Integrate an ODE using the adjoint method in the backward pass."""
        pass
