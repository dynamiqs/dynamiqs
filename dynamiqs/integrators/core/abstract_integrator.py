from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from ...gradient import Gradient
from ...method import Method
from ...result import Result, Saved, SolveSaved, StochasticSolveResult
from .interfaces import OptionsInterface

# _SavedT lets BaseIntegrator subclasses specify which concrete Saved type they work
# with (e.g. SolveSaved, PropagatorSaved). Making BaseIntegrator generic over
# _SavedT, methods like `result()` and `postprocess_saved()` gets per-subclass
# type signatures instead of accepting/returning the broad Saved base class.
_SavedT = TypeVar('_SavedT', bound=Saved)


class AbstractIntegrator(eqx.Module):
    """Abstract integrator.

    Any integrator should inherit from this class and implement the `run()` method
    to specify the main computationally intensive logic. This class is intentionally
    kept abstract to simplify the implementation of new integrators from scratch.
    """

    @abstractmethod
    def run(self) -> PyTree:
        pass


class BaseIntegrator(AbstractIntegrator, OptionsInterface, Generic[_SavedT]):
    """Integrator evolving an initial state over a set of times.

    This integrator evolves the initial pytree `y0` over a set of times specified by
    `ts`. It support multiple `method` and `gradient`, can be parameterized with
    `options`, and return a `result` object.
    """

    ts: Array
    y0: PyTree
    method: Method
    gradient: Gradient | None
    result_class: type[Result]

    @property
    def t0(self) -> Scalar:
        return self.ts[0] if self.options.t0 is None else jnp.asarray(self.options.t0)

    @property
    def t1(self) -> Scalar:
        return self.ts[-1]

    def result(self, saved: _SavedT, infos: PyTree | None = None) -> Result:
        return self.result_class(
            self.ts, self.method, self.gradient, self.options, saved, infos
        )


class StochasticBaseIntegrator(BaseIntegrator[SolveSaved]):
    """Integrator stochastically evolving an initial state over a set of times.

    In addition to `BaseIntegrator`, it includes a PRNG key for the stochastic
    evolution.
    """

    key: PRNGKeyArray
    result_class: type[StochasticSolveResult]

    def result(self, saved: SolveSaved, infos: PyTree | None = None) -> Result:
        ts = jnp.asarray(self.ts)  # todo: fix static tsave
        return self.result_class(
            ts, self.method, self.gradient, self.options, saved, infos, self.key
        )
