from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from ..._utils import _concatenate_sort
from ...gradient import Gradient
from ...result import (
    DSMESolveResult,
    MEPropagatorResult,
    MESolveResult,
    Result,
    Saved,
    SEPropagatorResult,
    SESolveResult,
)
from ...solver import Solver
from .interfaces import (
    DSMEInterface,
    MEInterface,
    OptionsInterface,
    SEInterface,
    SolveInterface,
)


class AbstractIntegrator(eqx.Module):
    """Abstract integrator.

    Any integrator should inherit from this class and implement the `run()` method
    to specify the main computationally intensive logic. This class is intentionally
    kept abstract to simplify the implementation of new integrators from scratch.
    """

    # subclasses should implement: run()

    @abstractmethod
    def run(self) -> PyTree:
        pass


class BaseIntegrator(AbstractIntegrator, OptionsInterface):
    """Integrator evolving an initial state over a set of times.

    This integrator evolves the initial pytree `y0` over a set of times specified by
    `ts`. It support multiple `solver` and `gradient`, can be parameterized with
    `options`, and return a `result` object.
    """

    # subclasses should implement: discontinuity_ts, run()

    RESULT_CLASS: ClassVar[Result]

    y0: PyTree
    ts: Array
    solver: Solver
    gradient: Gradient | None

    @property
    def t0(self) -> Scalar:
        return self.ts[0] if self.options.t0 is None else self.options.t0

    @property
    def t1(self) -> Scalar:
        return self.ts[-1]

    @property
    @abstractmethod
    def discontinuity_ts(self) -> Array | None:
        pass

    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        return self.RESULT_CLASS(
            self.ts, self.solver, self.gradient, self.options, saved, infos
        )


class SMEBaseIntegrator(BaseIntegrator):
    """Integrator stochastically evolving an initial state over a set of times, and
    saving measurement results at another set of times.

    In addition to `BaseIntegrator`, it includes a PRNG key `key` for the stochastic
    evolution.
    """

    # subclasses should implement: discontinuity_ts, run()

    key: PRNGKeyArray

    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        ts = jnp.asarray(self.ts)  # todo: fix static tsave
        return self.RESULT_CLASS(
            ts, self.solver, self.gradient, self.options, saved, infos, self.key
        )


class SEIntegrator(BaseIntegrator, SEInterface):
    """Integrator for the Schrödinger equation."""

    # subclasses should implement: run()

    @property
    def discontinuity_ts(self) -> Array | None:
        return self.H.discontinuity_ts


class MEIntegrator(BaseIntegrator, MEInterface):
    """Integrator for the Lindblad master equation."""

    # subclasses should implement: run()

    @property
    def discontinuity_ts(self) -> Array | None:
        ts = [x.discontinuity_ts for x in [self.H, *self.Ls]]
        return _concatenate_sort(*ts)


class DSMEIntegrator(SMEBaseIntegrator, DSMEInterface):
    """Integrator for the diffusive SME."""

    # subclasses should implement: run()

    @property
    def discontinuity_ts(self) -> Array | None:
        ts = [x.discontinuity_ts for x in [self.H, *self.Ls]]
        return _concatenate_sort(*ts)


class SEPropagatorIntegrator(SEIntegrator):
    """Integrator computing the propagator of the Schrödinger equation."""

    # subclasses should implement: run()

    RESULT_CLASS = SEPropagatorResult


class MEPropagatorIntegrator(MEIntegrator):
    """Integrator computing the propagator of the Lindblad master equation."""

    # subclasses should implement: run()

    RESULT_CLASS = MEPropagatorResult


class SESolveIntegrator(SEIntegrator, SolveInterface):
    """Integrator computing the time evolution of the Schrödinger equation."""

    # subclasses should implement: run()

    RESULT_CLASS = SESolveResult


class MESolveIntegrator(MEIntegrator, SolveInterface):
    """Integrator computing the time evolution of the Lindblad master equation."""

    # subclasses should implement: run()

    RESULT_CLASS = MESolveResult


class DSMESolveIntegrator(DSMEIntegrator, SolveInterface):
    """Integrator computing the time evolution of the diffusive SME."""

    # subclasses should implement: run()

    RESULT_CLASS = DSMESolveResult
