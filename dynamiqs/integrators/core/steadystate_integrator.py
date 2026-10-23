from __future__ import annotations

from collections.abc import Callable
from typing import cast

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from ...gradient import Gradient, Implicit
from ...method import DiffraxSteadyState
from ...options import Options
from ...qarrays.qarray import QArray
from ...qarrays.utils import asqarray
from ...result import MESteadyStateResult, SteadyStateSaved
from ...utils.general import expect, lindbladian
from ...utils.vectorization import unvectorize
from .abstract_integrator import AbstractIntegrator
from .diffrax_integrator import (
    AdaptiveStepInfos,
    FixedStepInfos,
    MESolveDiffraxIntegrator,
)


class SteadyStateInfos(eqx.Module):
    converged: Array
    time: Array
    steps: FixedStepInfos | AdaptiveStepInfos

    def __str__(self) -> str:
        if self.converged.ndim >= 1:
            nconverged = f'{self.converged.sum()}/{self.converged.size} converged'
            return f'{nconverged} | {self.steps}'
        status = 'converged' if self.converged else 'not converged'
        return f'{status} at t={self.time:.3g} | {self.steps}'


class MESteadyStateIntegrator(AbstractIntegrator):
    """Integrator evolving the Lindblad master equation until its steady state."""

    integrator: MESolveDiffraxIntegrator  # time integration from rho0
    method: DiffraxSteadyState
    gradient: Gradient | None
    options: Options
    Es: list[QArray] | None

    def run(self) -> MESteadyStateResult:
        if isinstance(self.gradient, Implicit | None):
            # the steady state does not depend on the path that reached it
            dynamic, static = eqx.partition(self.integrator, eqx.is_inexact_array)
            integrator = eqx.combine(jax.lax.stop_gradient(dynamic), static)
            rho, t, converged, stats = self._evolve(integrator)
            # the root is already known: `custom_root` only attaches its gradient
            x = jax.lax.custom_root(
                self._deflated_lindbladian(t), rho.to_jax(), lambda _, x: x, _solve
            )
            rho = asqarray(x, dims=rho.dims)
        else:
            rho, t, converged, stats = self._evolve(self.integrator)

        residual = _rms(lindbladian(self.integrator.H(t), self.integrator.L(t), rho))
        Esave = None
        if self.Es is not None:
            Esave = jnp.stack([expect(E, rho) for E in self.Es])
        saved = SteadyStateSaved(rho, None, Esave, residual)

        infos = SteadyStateInfos(converged, t, self.integrator.infos(stats))
        return MESteadyStateResult(
            None,  # ty: ignore[invalid-argument-type]
            self.method,
            self.gradient,
            self.options,
            saved,
            infos,
        )

    def _evolve(
        self, integrator: MESolveDiffraxIntegrator
    ) -> tuple[QArray, Array, Array, dict[str, Array]]:
        def cond_fn(t, y, args, **kwargs) -> Array:  # noqa: ANN001
            rho = unvectorize(y) if self.options.vectorized else y
            Lrho = lindbladian(integrator.H(t), integrator.L(t), rho)
            return _rms(Lrho) < self.method.atol + self.method.rtol * _rms(rho)

        solution = integrator.diffeqsolve(
            integrator.t0,
            integrator.t1,
            integrator.y0,
            dx.SaveAt(t1=True),
            event=dx.Event(cond_fn),
            throw=False,
        )
        ys = cast(QArray, solution.ys)
        rho = cast(QArray, ys[0])
        if self.options.vectorized:
            rho = unvectorize(rho)
        t = cast(Array, solution.ts)[0]
        return rho, t, cast(Array, solution.event_mask), solution.stats

    def _deflated_lindbladian(self, t: Array) -> Callable[[Array], Array]:
        # same unit-trace root as L(rho), but with Jacobian X -> L(X) - Tr[X] I/n,
        # invertible for a unique steady state (the Jacobian L is singular)
        H, Ls = self.integrator.H(t), self.integrator.L(t)
        n = H.shape[-1]

        def f(x: Array) -> Array:
            Lrho = lindbladian(H, Ls, asqarray(x, dims=H.dims)).to_jax()
            return Lrho + (1 - jnp.trace(x)) * jnp.eye(n) / n

        return f


def _solve(g: Callable[[Array], Array], y: Array) -> Array:
    # solve g(x) = y for the linear map g, with its (n^2, n^2) matrix
    n = y.shape[-1]
    G = jax.jacfwd(g, holomorphic=True)(y).reshape(n**2, n**2)
    return jnp.linalg.solve(G, y.reshape(n**2)).reshape(n, n)


def _rms(x: QArray) -> Array:
    return optx.rms_norm(x.to_jax())
