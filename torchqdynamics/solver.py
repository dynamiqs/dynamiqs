from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .options import Options
from .utils.solver_utils import bexpect
from .utils.tensor_types import TDOperator


class Solver(ABC):
    GRADIENT_ALG = ['autograd']

    def __init__(
        self,
        H: TDOperator,
        y0: Tensor,
        t_save: Tensor,
        exp_ops: Tensor,
        options: Options,
        gradient_alg: str | None,
        parameters: tuple[torch.nn.Parameter, ...] | None,
    ):
        """

        Args:
            H:
            y0: Initial quantum state, of shape `(..., m, n)`.
            t_save: Times for which results are saved.
            exp_ops:
            options:
            gradient_alg:
            parameters (tuple of nn.Parameter): Parameters w.r.t. compute the gradients.
        """
        # check that `t_save` is valid (it must be a non-empty 1D tensor sorted in
        # strictly ascending order and containing only positive values)
        if t_save.ndim != 1 or len(t_save) == 0:
            raise ValueError('Argument `t_save` must be a non-empty 1D tensor.')
        if not torch.all(torch.diff(t_save) > 0):
            raise ValueError(
                'Argument `t_save` must be sorted in strictly ascending order.'
            )
        if not torch.all(t_save >= 0):
            raise ValueError('Argument `t_save` must contain positive values only.')

        # check that the gradient algorithm is supported
        if gradient_alg is not None and gradient_alg not in self.GRADIENT_ALG:
            raise ValueError(
                f'Gradient algorithm {gradient_alg} is not defined or not yet'
                f' supported by this solver ({type(self)}).'
            )

        self.H = H
        self.y0 = y0
        self.t_save = t_save
        self.exp_ops = exp_ops
        self.options = options
        self.gradient_alg = gradient_alg
        self.parameters = parameters

        # initialize save tensors
        batch_sizes, (m, n) = self.y0.shape[:-2], self.y0.shape[-2:]

        if self.options.save_states:
            # y_save: (..., len(t_save), m, n)
            self.y_save = torch.zeros(
                *batch_sizes,
                len(self.t_save),
                m,
                n,
                dtype=self.y0.dtype,
                device=self.y0.device,
            )

        if len(self.exp_ops) > 0:
            # exp_save: (..., len(exp_ops), len(t_save))
            self.exp_save = torch.zeros(
                *batch_sizes,
                len(self.exp_ops),
                len(self.t_save),
                dtype=self.y0.dtype,
                device=self.y0.device,
            )
        else:
            self.exp_save = None

    def run(self):
        if self.gradient_alg is None:
            self.odeint_inplace()
        elif self.gradient_alg == 'autograd':
            self.odeint()

    def odeint_inplace(self):
        """Integrate a quantum ODE with an in-place solver.

        Simple solution for now so torch does not store gradients.
        TODO Implement a genuine in-place integrator.
        """
        with torch.no_grad():
            self.odeint()

    @abstractmethod
    def odeint(self):
        """Integrate a quantum ODE starting from an initial state.

        The ODE is solved from time `t=0.0` to `t=t_save[-1]`.
        """
        pass

    def save(self, i: int, y: Tensor):
        self._save_y(i, y)
        self._save_exp_ops(i, y)

    def _save_y(self, i: int, y: Tensor):
        if self.options.save_states:
            self.y_save[..., i, :, :] = y
        # otherwise only save the state if it is the final state
        elif i == len(self.t_save) - 1:
            self.y_save = y

    def _save_exp_ops(self, i: int, y: Tensor):
        if len(self.exp_ops) > 0:
            self.exp_save[..., i] = bexpect(self.exp_ops, y)
