from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple

import jax
from jax import Array
from jaxtyping import PyTree

from ..types import Scalar

Result = PyTree


class AbstractSolver:
    """Base abstract class for a generic solver.

    Evolve a quantum state `y` in time, from `ts[0]` to `ts[-1]`.

    The saved quantities returned by `run` may derive from the abstract state (e.g.
    expectation values of an operator), but not necessarily.
    """

    def __init__(self, ts: Array, y0: Array):
        self.ts = ts
        self.y0 = y0

    @abstractmethod
    def run(self) -> Result:
        # return a result object
        pass


class SolverState(NamedTuple):
    """Base class for the (hidden) state of a generic iterative solver.

    The solver state is passed on from one iteration to another. It is not the same as
    the quantum state `y`. The solver state gathers all needed elements to iterate the
    solver in a stateless fashion.
    """

    y: Array
    result: Result


class IterativeSolver(AbstractSolver):
    """Base abstract class for a generic iterative solver.

    Evolve a quantum state `y` iteratively in time. Results are saved for each time in
    `ts`.
    """

    def init_solver_state(self) -> SolverState:
        # initialise the hidden state of the solver
        pass

    @abstractmethod
    def step(self, t0: Scalar, t1: Scalar, solver_state: SolverState) -> SolverState:
        # make a step of the solver from t0 to t1
        pass

    @abstractmethod
    def save(self, t: Scalar, solver_state: SolverState) -> SolverState:
        pass

    def run(self) -> Result:
        # main solver loop
        solver_state = self.init_solver_state()

        # save initial state
        solver_state = self.save(self.ts[0], solver_state)

        def body_fun(i, solver_state):
            t0 = self.ts[i - 1]
            t1 = self.ts[i]
            # make a step
            solver_state = self.step(t0, t1, solver_state)
            # save
            solver_state = self.save(t1, solver_state)
            return solver_state

        solver_state = jax.lax.fori_loop(1, len(self.ts), body_fun, solver_state)

        return solver_state.result


class StatelessIterativeSolver(IterativeSolver):
    """Base abstract class for a stateless iterative solver.

    When the solver steps are independent we don't need to track the internal solver
    state, we simply use the state `y`.
    """

    @abstractmethod
    def _step(self, t0: Scalar, t1: Scalar, y: Array) -> Array:
        pass

    def step(self, t0: Scalar, t1: Scalar, solver_state: SolverState) -> SolverState:
        return solver_state._replace(y=self._step(t0, t1, solver_state.y))


class PropagatorSolver(StatelessIterativeSolver):
    @abstractmethod
    def forward(self, t: Scalar, delta_t: Scalar, y: Array) -> Array:
        pass

    def _step(self, t0: Scalar, t1: Scalar, y: Array) -> Array:
        delta_t = t1 - t0
        return self.forward(t0, delta_t, y)


class FixedStepSolver(StatelessIterativeSolver):
    """Naive implementation of a fixed step solver. For testing purposes only.

    This solver does not support reverse-mode autodiff.
    """

    @property
    @abstractmethod
    def dt(self) -> Scalar:
        pass

    @abstractmethod
    def forward(self, t: Scalar, y: Array) -> Array:
        pass

    def _step(self, t0: Scalar, t1: Scalar, y: Array) -> Array:
        def cond_fun(args):
            t, _ = args
            return t < t1

        def body_fun(args):
            t, y = args
            return t + self.dt, self.forward(t, y)

        _, y = jax.lax.while_loop(cond_fun, body_fun, (t0, y))

        return y
