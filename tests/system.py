from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from jax import Array
from jax import numpy as jnp
from jaxtyping import ArrayLike

import dynamiqs as dq
from dynamiqs.gradient import Gradient
from dynamiqs.result import Result
from dynamiqs.solver import Solver


class System(ABC):
    def __init__(self):
        self.n = None
        self.H = None
        self.Hb = None
        self.y0 = None
        self.y0b = None
        self.E = None

    @abstractmethod
    def tsave(self, n: int) -> Array:
        """Compute the save time array."""
        pass

    def state(self, t: float) -> Array:
        """Compute the exact state at a given time."""
        raise NotImplementedError

    def states(self, t: Array) -> Array:
        return jnp.stack([self.state(t_.item()) for t_ in t])

    def expect(self, t: float) -> Array:
        """Compute the exact (complex) expectation values at a given time."""
        raise NotImplementedError

    def expects(self, t: Array) -> Array:
        return jnp.stack([self.expect(t_.item()) for t_ in t]).swapaxes(0, 1)

    def loss_state(self, state: Array) -> Array:
        """Compute an example loss function from a given state."""
        return dq.expect(self.loss_op, state).real

    def grads_states(self, t: float) -> Array:
        """Compute the exact gradients of the example state loss function with respect
        to the system parameters.

        The returned array has shape _(num_params)_.
        """
        raise NotImplementedError

    def loss_expect(self, expect: Array) -> Array:
        """Compute example loss functions for each expectation values."""
        return jnp.stack(tuple(x.real for x in expect))

    def grads_expect(self, t: float) -> Array:
        """Compute the exact gradients of the example expectation values loss functions
        with respect to the system parameters.

        The returned array has shape _(nE, num_params)_.
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
