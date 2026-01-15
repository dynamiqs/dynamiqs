# ruff: noqa: ANN001, ANN201, ARG002
# we mostly ignore type hinting in this file for readability purposes

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import replace
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

    kraus_map: Callable[[RealScalarLike, RealScalarLike, Y], [Y, Y]]
    # should be defined as `kraus_map(t0, t1, y0) -> y1, error`

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
        y1, error = terms.term.kraus_map(t0, t1, y0)
        dense_info = dict(y0=y0, y1=y1)
        return y1, error, dense_info, None, dx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        pass

    def order(self, terms):
        return self._order


class AdaptiveRouchonDXSolver(dx.AbstractAdaptiveSolver, RouchonDXSolver):
    pass

def apply_nested_map(rho: QArray, Mss: Sequence[Sequence[QArray]]) -> QArray:
    """Applies the partial Kraus map defined by the operators
    Mss to the density matrix rho recursively.
    """
    res = rho
    for Ms in reversed(Mss):
        res = sum([M @ res @ M.dag() for M in Ms])
    return res

def compute_partial_S(rho: QArray, Mss: Sequence[Sequence[QArray]]) -> QArray:
    """Computes the corresponding operator S = Mk^† @ Mk for the Kraus operators Mss."""
    S = eye_like(rho)
    for Ms in Mss:
        S = sum([M.dag() @ S @ M for M in Ms])
    return S

def cholesky_normalize(
    Msss: Sequence[Sequence[Sequence[QArray]]], rho: QArray
) -> jax.Array:
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

    S = sum([compute_partial_S(rho, Mss) for Mss in Msss])
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


class MESolveFixedRouchonIntegrator(MESolveDiffraxIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using a
    fixed step Rouchon method.
    """

    @property
    def terms(self) -> dx.AbstractTerm:
        def kraus_map(t0, t1, y0):  # noqa: ANN202
            # The Rouchon update for a single loss channel is:
            #   rho_{k+1} = sum_k Mk @ rho_k @ Mk^†
            # See comment of `cholesky_normalize()` for the normalization.

            rho = y0
            t = t0
            dt = t1 - t0
            Msss = self._kraus_ops(t, dt)

            if self.method.normalize:
                rho = cholesky_normalize(Msss, rho)

            # for fixed step size, we return None for the error estimate
            return sum([apply_nested_map(rho, Mss) for Mss in Msss]), None

        return AbstractRouchonTerm(kraus_map)

    def _kraus_ops(self, t: float, dt: float) -> Sequence[QArray]:
        return self.Msss(self.H, self.L, t, dt, self.method.exact_expm)

    @staticmethod
    @abstractmethod
    def Msss(
        H: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: float,
        dt: float,
        exact_expm: bool,
    ) -> Sequence[QArray]:
        pass


class MESolveFixedRouchon1Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 1 method.
    """

    @staticmethod
    def Msss(
        H: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: float,
        dt: float,
        exact_expm: bool,
    ) -> Sequence[Sequence[Sequence[QArray]]]:        # M0 = I - (iH + 0.5 sum_k Lk^† @ Lk) dt
        Lmid = L(t + dt / 2)
        Hmid = H(t + dt / 2)
        # Mk = Lk sqrt(dt)
        LdL = sum([_L.dag() @ _L for _L in Lmid])
        Gmid = -1j * Hmid - 0.5 * LdL
        e1 = (dt * Gmid).expm() if exact_expm else _expm_taylor(dt * Gmid, 1)
        return [[[e1]] ,[[jnp.sqrt(dt) * _L for _L in Lmid]]]


mesolve_rouchon1_integrator_constructor = (
    lambda **kwargs: MESolveFixedRouchon1Integrator(
        **kwargs, diffrax_solver=RouchonDXSolver(1), fixed_step=True
    )
)


class MESolveFixedRouchon2Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 2 method.
    """

    @staticmethod
    def Msss(
        H: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: float,
        dt: float,
        exact_expm: bool,
    ) -> Sequence[Sequence[Sequence[QArray]]]:
        Lmid = L(t + dt/2)
        Hmid = H(t + dt/2)
        LdL = sum([_L.dag() @ _L for _L in Lmid])
        Gmid = -1j * Hmid - 0.5 * LdL
        e1 = (dt * Gmid).expm() if exact_expm else _expm_taylor(dt * Gmid, 2)
        J0 = [[[e1]]] # No jump kraus operator
        J1a = [[[jnp.sqrt(dt / 2) * e1 @ _L for _L in Lmid]]] # 1 jump a
        J1b = [[[jnp.sqrt(dt / 2) * _L @ e1 for _L in Lmid]]] # 1 jump b
        J2 = [[[jnp.sqrt(dt**2 / 2) * _L1 for _L1 in Lmid], Lmid]] # 2 jumps
        return [*J0, *J1a, *J1b, *J2]


class MESolveFixedRouchon3Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 3 method.
    """

    @staticmethod
    def Msss(
        H: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: float,
        dt: float,
        exact_expm: bool,
    ) -> Sequence[Sequence[Sequence[QArray]]]:
        Lmid = L(t + dt / 2)
        Hmid = H(t + dt / 2)
        LdL = sum([_L.dag() @ _L for _L in Lmid])
        Gmid = -1j * Hmid - 0.5 * LdL
        e1o3 = (dt / 3 * Gmid).expm() if exact_expm else _expm_taylor(dt / 3 * Gmid, 3)
        e2o3 = e1o3 @ e1o3
        e3o3 = e2o3 @ e1o3
        J0 = [[[e3o3]]]  # No jump kraus operator
        J1a = [[[jnp.sqrt(3 * dt / 4) * e1o3 @ _L @ e2o3 for _L in Lmid]]]  # 1 jump a
        J1b = [[[jnp.sqrt(dt / 4) * e3o3 @ _L for _L in Lmid]]]  # 1 jump b
        J2 = [[[jnp.sqrt(dt**2 / 2) * e1o3 @ _L1 for _L1 in Lmid], [e1o3 @ _L2 @ e1o3 for _L2 in Lmid]]]  # 2 jumps
        J3 = [[[jnp.sqrt(dt**3 / 6) * _L1 for _L1 in Lmid], Lmid, Lmid]]  # 3 jumps
        return [*J0, *J1a, *J1b, *J2, *J3]


class MESolveAdaptiveRouchonIntegrator(MESolveDiffraxIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using an
    adaptive Rouchon method.
    """

    @property
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        # todo: can we do better?
        stepsize_controller = super().stepsize_controller
        # fix incorrect default linear interpolation by stepping exactly at all times
        # in tsave, so interpolation is bypassed
        return replace(stepsize_controller, step_ts=self.ts)


class MESolveAdaptiveRouchon2Integrator(MESolveAdaptiveRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    adaptive Rouchon 1-2 method.
    """

    @property
    def terms(self) -> dx.AbstractTerm:
        def kraus_map(t0, t1, y0):  # noqa: ANN202
            rho = y0
            t = t0
            dt = t1 - t0

            L, H = self.L, self.H

            # === first order
            Msss_1 = MESolveFixedRouchon1Integrator.Msss(
                H, L, t, dt, self.method.exact_expm
            )
            rho_1 = cholesky_normalize(Msss_1, rho) if self.method.normalize else rho
            rho_1 = sum([apply_nested_map(rho_1, Mss) for Mss in Msss_1])

            # === second order
            Msss_2 = MESolveFixedRouchon2Integrator.Msss(
                H, L, t, dt, self.method.exact_expm
            )
            rho_2 = cholesky_normalize(Msss_2, rho) if self.method.normalize else rho
            rho_2 = sum([apply_nested_map(rho_2, Mss) for Mss in Msss_2])

            return rho_2, 0.5 * (rho_2 - rho_1)

        return AbstractRouchonTerm(kraus_map)


class MESolveAdaptiveRouchon3Integrator(MESolveAdaptiveRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    adaptive Rouchon 2-3 method.
    """

    @property
    def terms(self) -> dx.AbstractTerm:
        def kraus_map(t0, t1, y0):  # noqa: ANN202
            rho = y0
            t = t0
            dt = t1 - t0

            L, H = self.L, self.H

            # === second order
            Msss_2 = MESolveFixedRouchon2Integrator.Msss(
                H, L, t, dt, self.method.exact_expm
            )
            rho_2 = cholesky_normalize(Msss_2, rho) if self.method.normalize else rho
            rho_2 = sum([apply_nested_map(rho_2, Mss) for Mss in Msss_2])

            # === third order
            Msss_3 = MESolveFixedRouchon3Integrator.Msss(
                H, L, t, dt, self.method.exact_expm
            )
            rho_3 = cholesky_normalize(Msss_3, rho) if self.method.normalize else rho
            rho_3 = sum([apply_nested_map(rho_3, Mss) for Mss in Msss_3])
            return rho_3, 0.5 * (rho_3 - rho_2)

        return AbstractRouchonTerm(kraus_map)


def mesolve_rouchon2_integrator_constructor(**kwargs):
    """Factory function to create a Rouchon2 integrator."""
    if kwargs['method'].dt is not None:
        return MESolveFixedRouchon2Integrator(
            **kwargs, diffrax_solver=RouchonDXSolver(2), fixed_step=True
        )
    return MESolveAdaptiveRouchon2Integrator(
        **kwargs, diffrax_solver=AdaptiveRouchonDXSolver(2), fixed_step=False
    )


def mesolve_rouchon3_integrator_constructor(**kwargs):
    """Factory function to create a Rouchon3 integrator."""
    if kwargs['method'].dt is not None:
        return MESolveFixedRouchon3Integrator(
            **kwargs, diffrax_solver=RouchonDXSolver(3), fixed_step=True
        )
    return MESolveAdaptiveRouchon3Integrator(
        **kwargs, diffrax_solver=AdaptiveRouchonDXSolver(3), fixed_step=False
    )
