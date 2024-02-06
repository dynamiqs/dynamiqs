from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from jax import Array
from jax import numpy as jnp
from jaxtyping import PyTree

from dynamiqs.gradient import Gradient
from dynamiqs.result import Result
from dynamiqs.solver import Solver


class System(ABC):
    def __init__(self):
        self.n = None
        self.tsave = None
        self.params_default = None

    @abstractmethod
    def H(self, params: PyTree) -> Array:
        """Compute the Hamiltonian."""
        pass

    @abstractmethod
    def y0(self, params: PyTree) -> Array:
        """Compute the initial state."""
        pass

    @abstractmethod
    def E(self, params: PyTree) -> Array:
        """Compute the expectation value operators."""
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
        raise NotImplementedError

    def grads_states(self, t: float) -> PyTree:
        """Compute the exact gradients of the example state loss function with respect
        to the system parameters."""
        raise NotImplementedError

    def loss_expect(self, expect: Array) -> Array:
        """Compute example loss functions for each expectation values."""
        return expect.real

    def grads_expect(self, t: float) -> PyTree:
        """Compute the exact gradients of the example expectation values loss functions
        with respect to the system parameters."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        solver: Solver,
        *,
        gradient: Gradient | None = None,
        options: dict[str, Any] | None = None,
        params: PyTree | None = None,
    ) -> Result:
        pass
