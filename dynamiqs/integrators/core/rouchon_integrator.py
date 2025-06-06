# ruff: noqa: ANN001, ANN201, ARG002
# we mostly ignore type hinting in this file for readability purposes

from __future__ import annotations

from abc import abstractmethod
from itertools import product

import diffrax as dx
import jax
import jax.numpy as jnp
from diffrax._custom_types import RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation

from ...qarrays.qarray import QArray
from ...utils.operators import eye_like
from .diffrax_integrator import MESolveDiffraxIntegrator


class AbstractRouchonTerm(dx.AbstractTerm):
    # this class bypasses the typical Diffrax term implementation, as Rouchon schemes
    # don't match the vf/contr/prod structure

    kraus_map: callable[[RealScalarLike, RealScalarLike, Y], Y]
    # should be defined as `kraus_map(t0, t1, y0) -> y1`

    def vf(self, t, y, args):
        pass

    def contr(self, t0, t1, **kwargs):
        pass

    def prod(self, vf, control):
        pass


class RouchonDXSolver(dx.AbstractSolver):
    _order: int
    term_structure = AbstractRouchonTerm
    interpolation_cls = LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        pass

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        y1 = terms.term.kraus_map(t0, t1, y0)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, dx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        pass

    def order(self, terms):
        return self._order


def cholesky_normalize(Ms: list[QArray], rho: QArray) -> jax.Array:
    # To normalize the scheme, we compute
    #   S = sum_k Mk^† @ Mk
    # and replace
    #   Mk by ~Mk = Mk @ S^{-1/2}
    # such that
    #   sum_k ~Mk^† @ ~Mk = S^{†(-1/2)} @ (sum_k Mk^† @ Mk) @ S^{-1/2}
    #                   = S^{†(-1/2)} @ S @ S^{-1/2}
    #                   = I
    # To (i) keep sparse matrices and (ii) have a generic implementation that also
    # works for time-dependent systems, we use a Cholesky decomposition at each step
    # instead of computing S^{-1/2} explicitly. We write S = T @ T^† with T lower
    # triangular, and we replace
    #   Mk by ~Mk = Mk @ S^{-1/2}
    # such that
    #   #   sum_k ~Mk^† @ ~Mk = T^{-1} @ (sum_k Mk^† @ Mk) @ T^{†(-1)}
    #                       = T^{-1} @ T @ T^† @ T^{(-1)}
    #                       = I
    # In practice we directly replace rho_k by T^{†(-1)} @ rho_k @ T^{-1} instead of
    # computing all ~Mks.

    S = sum([M.dag() @ M for M in Ms])
    T = jnp.linalg.cholesky(S.to_jax())  # T lower triangular

    # we want T^{†(-1)} @ y0 @ T^{-1}
    rho = rho.to_jax()
    # solve T^† @ x = rho => x = T^{†(-1)} @ rho
    rho = jax.lax.linalg.triangular_solve(
        T, rho, lower=True, transpose_a=True, conjugate_a=True
    )
    # solve x @ T = rho => x = rho @ T^{-1}
    return jax.lax.linalg.triangular_solve(T, rho, lower=True, left_side=True)


def _expm_taylor(A: QArray, order: int) -> QArray:
    I = eye_like(A)
    out = I
    powers_of_A = I
    for i in range(1, order + 1):
        powers_of_A = A @ powers_of_A
        out += 1 / jax.scipy.special.factorial(i) * powers_of_A

    return out


class MESolveRouchonIntegrator(MESolveDiffraxIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    Rouchon method.
    """

    @property
    def terms(self) -> dx.AbstractTerm:
        def kraus_map(t0, t1, y0):  # noqa: ANN202
            # The Rouchon update for a single loss channel is:
            #   rho_{k+1} = sum_k Mk @ rho_k @ Mk^†
            # See comment of `cholesky_normalize()` for the normalisation.

            rho = y0
            t = (t0 + t1) / 2
            dt = t1 - t0
            Ms = self._kraus_ops(t, dt)

            if self.method.normalize:
                rho = cholesky_normalize(Ms, rho)

            return sum([M @ rho @ M.dag() for M in Ms])

        return AbstractRouchonTerm(kraus_map)

    @abstractmethod
    def _kraus_ops(self, t: float, dt: float) -> list[QArray]:
        pass


class MESolveRouchon1Integrator(MESolveRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    Rouchon 1 method.
    """

    def _kraus_ops(self, t: float, dt: float) -> list[QArray]:
        # M0 = I - (iH + 0.5 sum_k Lk^† @ Lk) dt
        # Mk = Lk sqrt(dt)
        L, H = self.L(t), self.H(t)
        LdL = sum([_L.dag() @ _L for _L in L])
        G = -1j * H - 0.5 * LdL
        e1 = (dt * G).expm() if self.method.exact_expm else _expm_taylor(dt * G, 1)
        return [e1] + [jnp.sqrt(dt) * _L for _L in L]


mesolve_rouchon1_integrator_constructor = lambda **kwargs: MESolveRouchon1Integrator(
    **kwargs, diffrax_solver=RouchonDXSolver(1), fixed_step=True
)


class MESolveRouchon2Integrator(MESolveRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    Rouchon 2 method.
    """

    def _kraus_ops(self, t: float, dt: float) -> list[QArray]:
        L, H = self.L(t), self.H(t)
        LdL = sum([_L.dag() @ _L for _L in L])
        G = -1j * H - 0.5 * LdL
        e1 = (dt * G).expm() if self.method.exact_expm else _expm_taylor(dt * G, 2)

        return (
            [e1]
            + [jnp.sqrt(dt / 2) * e1 @ _L for _L in L]
            + [jnp.sqrt(dt / 2) * _L @ e1 for _L in L]
            + [jnp.sqrt(dt**2 / 2) * _L1 @ _L2 for _L1, _L2 in product(L, L)]
        )


mesolve_rouchon2_integrator_constructor = lambda **kwargs: MESolveRouchon2Integrator(
    **kwargs, diffrax_solver=RouchonDXSolver(2), fixed_step=True
)


class MESolveRouchon3Integrator(MESolveRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    Rouchon 3 method.
    """

    def _kraus_ops(self, t: float, dt: float) -> list[QArray]:
        L, H = self.L(t), self.H(t)
        LdL = sum([_L.dag() @ _L for _L in L])
        G = -1j * H - 0.5 * LdL
        e1o3 = (
            (dt / 3 * G).expm()
            if self.method.exact_expm
            else _expm_taylor(dt / 3 * G, 3)
        )
        e2o3 = e1o3 @ e1o3
        e3o3 = e2o3 @ e1o3

        return (
            [e3o3]
            + [jnp.sqrt(3 * dt / 4) * e1o3 @ _L @ e2o3 for _L in L]
            + [jnp.sqrt(dt / 4) * e3o3 @ _L for _L in L]
            + [
                jnp.sqrt(dt**2 / 2) * e1o3 @ _L1 @ e1o3 @ _L2 @ e1o3
                for _L1, _L2 in product(L, L)
            ]
            + [
                jnp.sqrt(dt**3 / 6) * _L1 @ _L2 @ _L3
                for _L1, _L2, _L3 in product(L, L, L)
            ]
        )


mesolve_rouchon3_integrator_constructor = lambda **kwargs: MESolveRouchon3Integrator(
    **kwargs, diffrax_solver=RouchonDXSolver(3), fixed_step=True
)
