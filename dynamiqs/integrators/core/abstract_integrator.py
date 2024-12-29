from __future__ import annotations

from abc import abstractmethod
from enum import Enum

import equinox as eqx
from jax import Array
from jaxtyping import PyTree, Scalar

from ...gradient import Gradient
from ...result import (
    MEPropagatorResult,
    MESolveResult,
    Result,
    Saved,
    SEPropagatorResult,
    SESolveResult,
)
from ...solver import Solver
from .interfaces import OptionsInterface, TimeInterface


class SOLVER_FUNCTION(Enum):
    SESOLVE = 0
    MESOLVE = 1
    SEPROPAGATOR = 2
    MEPROPAGATOR = 3
    FLOQUET = 4


class AbstractIntegrator(eqx.Module):
    """Abstract integrator.

    Any integrator should inherit from this class and implement the `run()` method
    to specify the main computationally intensive logic. This class is intentionally
    kept abstract to simplify the implementation of new integrators from scratch.
    """

    @abstractmethod
    def run(self) -> PyTree:
        pass


class BaseIntegrator(AbstractIntegrator, OptionsInterface, TimeInterface):
    """Integrator evolving an initial state over a set of times.

    This integrator evolves the initial pytree `y0` over a set of times specified by
    `ts`. It support multiple `solver` and `gradient`, can be parameterized with
    `options`, and return a `result` object.
    """

    ts: Array
    y0: PyTree
    solver: Solver
    gradient: Gradient | None
    solver_function: str

    @property
    def t0(self) -> Scalar:
        return self.ts[0] if self.options.t0 is None else self.options.t0

    @property
    def t1(self) -> Scalar:
        return self.ts[-1]

    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        result_classes = {
            SOLVER_FUNCTION.SESOLVE: SESolveResult,
            SOLVER_FUNCTION.MESOLVE: MESolveResult,
            SOLVER_FUNCTION.SEPROPAGATOR: SEPropagatorResult,
            SOLVER_FUNCTION.MEPROPAGATOR: MEPropagatorResult,
        }
        result_class = result_classes[self.solver_function]
        return result_class(
            self.ts, self.solver, self.gradient, self.options, saved, infos
        )
