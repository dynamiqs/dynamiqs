from __future__ import annotations

import functools
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ..options import Options
from ..utils.solver_utils import bexpect
from ..utils.td_tensor import TDTensor


def depends_on_H(func):
    """Handles caching for functions that only depend on the Hamiltonian. The functions
    must take as single argument the time and be decorated by `@depends_on_H`.

    Example:
        >>> @depends_on_H
        >>> def H_squared(self, t):
        >>>     return self.H(t) @ self.H(t)

    Warning:
        Caching is only checked through the time variable `t`. If different arguments
        are passed to the function for the same time stamp, the cached object will be
        returned instead of a newly computed one.
    """

    @functools.wraps(func)
    def wrapper(instance, t, *args, **kwargs):
        if func.__name__ not in instance._cache or (
            t != instance._cache[func.__name__][0] and instance.H.has_changed(t)
        ):
            instance._cache[func.__name__] = t, func(instance, t, *args, **kwargs)
        return instance._cache[func.__name__][1]

    return wrapper


class Solver(ABC):
    def __init__(
        self,
        H: TDTensor,
        y0: Tensor,
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

        self.H = H
        self.y0 = y0
        self.t_save = t_save
        self.exp_ops = exp_ops
        self.options = options
        self._cache = {}

        self.save_counter = 0

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
        if self.options.gradient_alg is None:
            self.run_nograd()

    @abstractmethod
    def run_nograd(self):
        pass

    def next_tsave(self) -> float:
        return self.t_save[self.save_counter]

    def save(self, y: Tensor):
        self._save_y(y)
        self._save_exp_ops(y)
        self.save_counter += 1

    def _save_y(self, y: Tensor):
        if self.options.save_states:
            self.y_save[..., self.save_counter, :, :] = y
        # otherwise only save the state if it is the final state
        elif self.save_counter == len(self.t_save) - 1:
            self.y_save = y

    def _save_exp_ops(self, y: Tensor):
        if len(self.exp_ops) > 0:
            self.exp_save[..., self.save_counter] = bexpect(self.exp_ops, y)


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
