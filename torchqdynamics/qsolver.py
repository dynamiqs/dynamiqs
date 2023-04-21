from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from .solver_options import SolverOption
from .solver_utils import bexpect


class QSolver(ABC):
    def __init__(
        self,
        options: SolverOption,
        y0: Tensor,
        exp_ops: Tensor,
        t_save: Tensor,
        gradient_alg: Literal['autograd', 'adjoint'] | None,
        parameters: tuple[nn.Parameter, ...] | None,
    ):
        self.options = options
        self.y0 = y0
        self.exp_ops = exp_ops
        self.t_save = t_save
        self.gradient_alg = gradient_alg
        self.parameters = parameters

        self.save_counter = 0

        # initialize save tensors
        batch_sizes, (m, n) = y0.shape[:-2], y0.shape[-2:]

        if self.options.save_states:
            # y_save: (..., len(t_save), m, n)
            self.y_save = torch.zeros(
                *batch_sizes, len(self.t_save), m, n, dtype=y0.dtype, device=y0.device
            )

        if len(self.exp_ops) > 0:
            # exp_save: (..., len(exp_ops), len(t_save))
            self.exp_save = torch.zeros(
                *batch_sizes,
                len(self.exp_ops),
                len(self.t_save),
                dtype=y0.dtype,
                device=y0.device,
            )
        else:
            self.exp_save = torch.empty(
                *batch_sizes, len(self.exp_ops), dtype=y0.dtype, device=y0.device
            )

    def next_tsave(self) -> float:
        return self.t_save[self.save_counter]

    def _save_y(self, y: Tensor):
        if self.options.save_states:
            self.y_save[..., self.save_counter, :, :] = y

    def _save_exp_ops(self, y: Tensor):
        if len(self.exp_ops) > 0:
            self.exp_save[..., self.save_counter] = bexpect(self.exp_ops, y)

    def save(self, y: Tensor):
        self._save_y(y)
        self._save_exp_ops(y)
        self.save_counter += 1

    def save_final(self, y: Tensor):
        if not self.options.save_states:
            self.y_save = y

    @abstractmethod
    def run(self):
        pass
