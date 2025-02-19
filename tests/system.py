from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from dynamiqs import QArray, stack
from dynamiqs.gradient import Gradient
from dynamiqs.method import Method
from dynamiqs.options import Options
from dynamiqs.result import Result
from dynamiqs.time_qarray import TimeQArray


class System(ABC):
    def __init__(self):
        self.n = None
        self.tsave = None
        self.params_default = None

    @abstractmethod
    def H(self, params: PyTree) -> QArray | TimeQArray:
        """Compute the Hamiltonian."""

    @abstractmethod
    def y0(self, params: PyTree) -> QArray:
        """Compute the initial state."""

    @abstractmethod
    def Es(self, params: PyTree) -> list[QArray]:
        """Compute the expectation value operators."""

    def state(self, t: float) -> QArray:
        """Compute the exact state at a given time."""
        raise NotImplementedError

    def states(self, t: Array) -> QArray:
        return stack([self.state(t_.item()) for t_ in t])

    def expect(self, t: float) -> Array:
        """Compute the exact (complex) expectation values at a given time."""
        raise NotImplementedError

    def expects(self, t: Array) -> Array:
        return jnp.stack([self.expect(t_.item()) for t_ in t]).swapaxes(0, 1)

    def loss_state(self, state: QArray) -> Array:
        """Compute an example loss function from a given state."""
        raise NotImplementedError

    def grads_states(self, t: float) -> PyTree:
        """Compute the exact gradients of the example state loss function with respect
        to the system parameters.
        """
        raise NotImplementedError

    def loss_expect(self, expect: Array) -> Array:
        """Compute example loss functions for each expectation values."""
        return expect.real

    def grads_expect(self, t: float) -> PyTree:
        """Compute the exact gradients of the example expectation values loss functions
        with respect to the system parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        method: Method,
        *,
        gradient: Gradient | None = None,
        options: Options = Options(),  # noqa: B008
        params: PyTree | None = None,
    ) -> Result:
        pass
