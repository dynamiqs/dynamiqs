# ruff: noqa: ANN001, ANN201, ARG002
# we mostly ignore type hinting in this file for readability purposes

from __future__ import annotations

import diffrax as dx
import jax
import jax.numpy as jnp
from diffrax._custom_types import RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation

from ...qarrays.qarray import QArray
from ...utils.general import dag
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


class Rouchon1DXSolver(RouchonDXSolver):
    def order(self, terms):
        return 1


def cholesky_normalize(M0: QArray, LdL: QArray, dt: float, rho: QArray) -> jax.Array:
    # To normalize the scheme, we compute S = M0d @ M0 + M1d @ M1 and replace
    #   M0 by ~M0 = M0 @ S^{-1/2}
    #   M1 by ~M1 = M1 @ S^{-1/2}
    # such that
    #   ~M0d @ ~M0 + ~M1d @ ~M1 = Sd^{-1/2} (M0d @ M0 + M1d @ M1) S^{-1/2}
    #                           = Sd^{-1/2} S S^{-1/2}
    #                           = I
    #
    # For efficiency, we use a Cholesky decomposition instead of computing
    # S^{-1/2} explicitly. We write S = T @ Td with T lower triangular, and we
    # replace
    #   M0 by ~M0 = M0 @ Td^{-1}
    #   M1 by ~M1 = M1 @ Td^{-1}
    # such that
    #   ~M0d @ ~M0 + ~M1d @ ~M1 = T^{-1} @ (M0d @ M0 + M1d @ M1) @ Td^{-1}
    #                           = T^{-1} @ T @ Td @ Td^{-1}
    #                           = I
    #
    # In practice we directly replace rho_k by Td^{-1} @ rho_k @ T^{-1/2}
    # instead of computing all ~Mks.

    # dt may be a traced array, so we need to be careful with the sum
    S = M0.dag() @ M0 + (LdL * dt if LdL != 0.0 else 0.0)

    T = jnp.linalg.cholesky(S.to_jax())  # T lower triangular

    # we want Td^{-1} @ y0 @ T^{-1}
    rho = rho.to_jax()
    # solve Td @ x = rho => x = Td^{-1} @ rho
    rho = jax.lax.linalg.triangular_solve(
        T, rho, lower=True, transpose_a=True, conjugate_a=True
    )
    # solve x @ T = rho => x = rho @ T^{-1}
    return jax.lax.linalg.triangular_solve(T, rho, lower=True, left_side=True)


class MESolveRouchon1Integrator(MESolveDiffraxIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    Rouchon 1 method.
    """

    @property
    def terms(self) -> dx.AbstractTerm:
        def kraus_map(t0, t1, y0):  # noqa: ANN202
            # The Rouchon update for a single loss channel is:
            #   rho_{k+1} = M0 @ rho_k @ M0d + M1 @ rho_k @ M1d
            # with
            #   M0 = I - (iH + 0.5 Ld @ L) dt
            #   M1 = L sqrt(dt)
            #
            # See comment of `cholesky_normalize()` for the normalisation.

            delta_t = t1 - t0
            rho = y0
            L, H = self.L(t0), self.H(t0)
            I = eye_like(H)
            LdL = sum([_L.dag() @ _L for _L in L])

            M0 = I - (1j * H + 0.5 * LdL) * delta_t
            Ms = [jnp.sqrt(delta_t) * _L for _L in L]

            if self.method.normalize:
                rho = cholesky_normalize(M0, LdL, delta_t, rho)

            return M0 @ rho @ dag(M0) + sum([_M @ rho @ dag(_M) for _M in Ms])

        return AbstractRouchonTerm(kraus_map)


mesolve_rouchon1_integrator_constructor = lambda **kwargs: MESolveRouchon1Integrator(
    **kwargs, diffrax_solver=Rouchon1DXSolver(), fixed_step=True
)
