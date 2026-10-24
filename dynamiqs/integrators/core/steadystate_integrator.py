from __future__ import annotations

from collections.abc import Callable
from typing import cast

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from ..._utils import cdtype
from ...gradient import Gradient, Implicit
from ...method import DiffraxSteadyState, LyapunovGMRES
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
from .gmres import gmres


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


class GMRESInfos(eqx.Module):
    converged: Array
    niter: Array

    def __str__(self) -> str:
        if self.converged.ndim >= 1:
            return f'{self.converged.sum()}/{self.converged.size} converged'
        status = 'converged' if self.converged else 'not converged'
        return f'{status} after {self.niter} GMRES cycles'


class MESteadyStateGMRESIntegrator(AbstractIntegrator):
    """Integrator solving the steady-state equation with preconditioned GMRES."""

    H: QArray
    Ls: list[QArray]
    rho0: QArray
    method: LyapunovGMRES
    gradient: Gradient | None
    options: Options
    Es: list[QArray] | None

    def run(self) -> MESteadyStateResult:
        n = self.H.shape[-1]
        I = jnp.eye(n, dtype=cdtype()).reshape(-1)  # vec(I)
        rho0 = self.rho0.to_jax().reshape(-1).astype(I.dtype)
        M = self._preconditioner(I)

        # L + |I><I| is invertible for a unique steady state, unlike L, and
        # (L + |I><I|)(rho) = I at the unit-trace steady state
        def A(x: Array) -> Array:
            rho = asqarray(x.reshape(n, n), dims=self.H.dims)
            Lrho = lindbladian(self.H, self.Ls, rho).to_jax().reshape(-1)
            return Lrho + (I @ x) * I

        def solve(A, b, M, x0):  # noqa: ANN001, ANN202
            # a zero b, e.g. a cotangent without sensitivity, would divide by zero
            is_zero = jnp.all(b == 0)
            b = jnp.where(is_zero, I, b)
            # atol scales with b, for the tangent solves
            atol = self.method.atol * optx.rms_norm(b) / optx.rms_norm(I)
            rtol = self.method.rtol
            stop = lambda x: optx.rms_norm(A(x) - b) < atol + rtol * optx.rms_norm(x)
            x, niter, converged = gmres(
                A,
                M,
                b,
                x0,
                stop,
                self.method.krylov_size,
                self.method.recycling,
                self.method.max_iter,
            )
            return jnp.where(is_zero, 0, x), (niter, converged)

        def forward_solve(A, b):  # noqa: ANN001, ANN202
            # the solution has trace Tr[b]/n: 1 for the steady state, 0 for a tangent
            return solve(A, b, M, rho0 * (I @ b) / n)

        def transpose_solve(AT, b):  # noqa: ANN001, ANN202
            MT = jax.linear_transpose(M, b)
            return solve(AT, b, lambda x: MT(x)[0], jnp.zeros_like(b))

        # implicit differentiation, with GMRES solves of A and A^T for the gradient
        x, (niter, converged) = jax.lax.custom_linear_solve(
            A, I, forward_solve, transpose_solve, has_aux=True
        )
        X = x.reshape(n, n)
        rho = 0.5 * (X + X.conj().mT)
        rho = asqarray(rho / jnp.trace(rho), dims=self.H.dims)

        residual = _rms(lindbladian(self.H, self.Ls, rho))
        Esave = None
        if self.Es is not None:
            Esave = jnp.stack([expect(E, rho) for E in self.Es])
        saved = SteadyStateSaved(rho, None, Esave, residual)

        infos = GMRESInfos(converged, niter)
        return MESteadyStateResult(
            None,  # ty: ignore[invalid-argument-type]
            self.method,
            self.gradient,
            self.options,
            saved,
            infos,
        )

    def _preconditioner(self, I: Array) -> Callable[[Array], Array]:
        # M ~ (L + |I><I|)^-1 from the no-jump part S(X) = Hnh X + X Hnh^† of
        # L(X) = S(X) + sum_k L_k X L_k^†, with S^-1 in the eigenbasis of Hnh and
        # Sherman-Morrison for |I><I|
        n = self.H.shape[-1]
        LdL = sum([_L.dag() @ _L for _L in self.Ls])
        Hnh = -1j * self.H - 0.5 * LdL
        # M changes the convergence of GMRES, not its solution
        Hnh = jax.lax.stop_gradient(Hnh.to_jax())
        e, W = jnp.linalg.eig(Hnh)
        Winv = jnp.linalg.inv(W)
        e_sum = e[:, None] + e[None, :].conj()

        def Sinv(x: Array) -> Array:
            Y = Winv @ x.reshape(n, n) @ Winv.conj().mT
            return (W @ (Y / e_sum) @ W.conj().mT).reshape(-1)

        Sinv_I = Sinv(I)

        def M(x: Array) -> Array:
            Sinv_x = Sinv(x)
            return Sinv_x - (I @ Sinv_x) / (1 + I @ Sinv_I) * Sinv_I

        return M


def _solve(g: Callable[[Array], Array], y: Array) -> Array:
    # solve g(x) = y for the linear map g, with its (n^2, n^2) matrix
    n = y.shape[-1]
    G = jax.jacfwd(g, holomorphic=True)(y).reshape(n**2, n**2)
    return jnp.linalg.solve(G, y.reshape(n**2)).reshape(n, n)


def _rms(x: QArray) -> Array:
    return optx.rms_norm(x.to_jax())
