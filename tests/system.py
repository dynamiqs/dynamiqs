from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import Any

import torch
from torch import Tensor

import dynamiqs as dq
from dynamiqs.gradient import Gradient
from dynamiqs.solver import Solver
from dynamiqs.solvers.result import Result
from dynamiqs.utils.tensor_types import ArrayLike


class System(ABC):
    def __init__(self):
        self.n = None
        self.H = None
        self.H_batched = None
        self.y0 = None
        self.y0_batched = None
        self.exp_ops = None

    @abstractproperty
    def _state_shape(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def tsave(self, n: int) -> Tensor:
        """Compute the save time tensor."""
        pass

    def state(self, t: float) -> Tensor:
        """Compute the exact state at a given time."""
        raise NotImplementedError

    def states(self, t: Tensor) -> Tensor:
        return torch.stack([self.state(t_.item()) for t_ in t])

    def expect(self, t: float) -> Tensor:
        """Compute the exact (complex) expectation values at a given time."""
        raise NotImplementedError

    def expects(self, t: Tensor) -> Tensor:
        return torch.stack([self.expect(t_.item()) for t_ in t]).swapaxes(0, 1)

    def loss_state(self, state: Tensor) -> Tensor:
        """Compute an example loss function from a given state."""
        return dq.expect(self.loss_op, state).real

    def grads_states(self, t: float) -> Tensor:
        """Compute the exact gradients of the example state loss function with respect
        to the system parameters.

        The returned tensor has shape _(num_params)_.
        """
        raise NotImplementedError

    def loss_expect(self, expect: Tensor) -> Tensor:
        """Compute example loss functions for each expectation values."""
        return torch.stack(tuple(x.real for x in expect))

    def grads_expect(self, t: float) -> Tensor:
        """Compute the exact gradients of the example expectation values loss functions
        with respect to the system parameters.

        The returned tensor has shape _(num_exp_ops, num_params)_.
        """
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        tsave: ArrayLike,
        solver: Solver,
        *,
        gradient: Gradient | None = None,
        options: dict[str, Any] | None = None,
    ) -> Result:
        pass
