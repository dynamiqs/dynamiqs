# ruff: noqa: ANN001, ANN201, ARG002
# we mostly ignore type hinting in this file for readability purposes

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import reduce
from itertools import product

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax._custom_types import RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation

from ...qarrays.qarray import QArray
from ...result import MESolveResult
from ...utils.operators import eye_like
from .diffrax_integrator import MESolveDiffraxIntegrator


class AbstractRouchonTerm(dx.AbstractTerm):
    # this class bypasses the typical Diffrax term implementation, as Rouchon schemes
    # don't match the vf/contr/prod structure

    rouchon_step: Callable[[RealScalarLike, RealScalarLike, Y], [Y, Y]]
    # should be defined as `rouchon_step(t0, t1, y0) -> y1, error`

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
        y1, error = terms.term.rouchon_step(t0, t1, y0)
        dense_info = dict(y0=y0, y1=y1)
        return y1, error, dense_info, None, dx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        pass

    def order(self, terms):
        return self._order


class AdaptiveRouchonDXSolver(dx.AbstractAdaptiveSolver, RouchonDXSolver):
    pass


class KrausChannel(eqx.Module):
    operators: list[QArray]

    def __call__(self, rho) -> QArray:
        return sum([M @ rho @ M.dag() for M in self.operators])

    def apply_conjugate(self, op) -> QArray:
        # computes S = sum(M† op M) for the channel
        return sum([M.dag() @ op @ M for M in self.operators])

    def S(self) -> QArray:
        # computes S = sum(M†M) for the channel
        return sum([M.dag() @ M for M in self.operators])

    def get_kraus_operators(self) -> list[QArray]:
        # returns the list of Kraus operators in the channel.
        return self.operators


class NestedKrausChannel(eqx.Module):
    channels: list[KrausChannel]

    def __init__(self, *channels: KrausChannel):
        self.channels = list(channels)

    def __call__(self, rho) -> QArray:
        for channel in reversed(self.channels):
            rho = channel(rho)
        return rho

    def S(self) -> QArray:
        # computes S = sum(M†M) for the nested channel
        if len(self.channels) == 0:
            return eye_like(self.channels[0].operators[0])

        out = self.channels[0].S()
        for channel in self.channels[1:]:
            out = channel.apply_conjugate(out)
        return out

    def get_kraus_operators(self) -> list[QArray]:
        # returns the list of all Kraus operators in the nested channel.
        return [
            reduce(lambda a, b: a @ b, x)  # product of all the operators
            for x in product(*[ch.operators for ch in self.channels])
        ]


class KrausMap(eqx.Module):
    channels: list[NestedKrausChannel | KrausChannel]

    def __init__(self, *channels: NestedKrausChannel | KrausChannel):
        self.channels = list(channels)

    def __call__(self, rho) -> QArray:
        return sum([channel(rho) for channel in self.channels])

    def S(self) -> QArray:
        # computes S = sum(M†M) for the full map
        return sum([channel.S() for channel in self.channels])

    def get_kraus_operators(self) -> list[QArray]:
        # returns the list of all Kraus operators in the map.
        return [op for c in self.channels for op in c.get_kraus_operators()]


def cholesky_normalize(kraus_map: KrausMap, rho: QArray) -> jax.Array:
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

    S = kraus_map.S()
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
        def rouchon_step(t0, t1, y0):  # noqa: ANN202
            # The Rouchon update for a single loss channel is:
            #   rho_{k+1} = sum_k Mk @ rho_k @ Mk^†
            # See comment of `cholesky_normalize()` for the normalization.

            rho = y0
            t = (t0 + t1) / 2
            dt = t1 - t0
            kraus_map = self._build_kraus_map(t, dt)

            if self.method.normalize:
                rho = cholesky_normalize(kraus_map, rho)

            # for fixed step size, we return None for the error estimate
            return kraus_map(rho), None

        return AbstractRouchonTerm(rouchon_step)

    def _build_kraus_map(self, t: float, dt: float) -> KrausMap:
        L, H = self.L(t), self.H(t)
        return self.build_kraus_map(H, L, dt, self.method.exact_expm)

    @staticmethod
    @abstractmethod
    def build_kraus_map(
        H: QArray, L: Sequence[QArray], dt: float, exact_expm: bool
    ) -> KrausMap:
        pass


class MESolveFixedRouchon1Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 1 method.
    """

    @staticmethod
    def build_kraus_map(
        H: QArray, L: Sequence[QArray], dt: float, exact_expm: bool
    ) -> KrausMap:
        LdL = sum([_L.dag() @ _L for _L in L])
        G = -1j * H - 0.5 * LdL
        e1 = (dt * G).expm() if exact_expm else _expm_taylor(dt * G, 1)
        channel = KrausChannel([e1] + [jnp.sqrt(dt) * _L for _L in L])
        return KrausMap(channel)


mesolve_rouchon1_integrator_constructor = lambda **kwargs: (
    MESolveFixedRouchon1Integrator(
        **kwargs,
        diffrax_solver=RouchonDXSolver(1),
        fixed_step=True,
        result_class=MESolveResult,
    )
)


class MESolveFixedRouchon2Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 2 method.
    """

    @staticmethod
    def build_kraus_map(
        H: QArray, L: Sequence[QArray], dt: float, exact_expm: bool
    ) -> KrausMap:
        LdL = sum([_L.dag() @ _L for _L in L])
        G = -1j * H - 0.5 * LdL
        e1 = (dt * G).expm() if exact_expm else _expm_taylor(dt * G, 2)
        channel_1 = KrausChannel(
            [e1]
            + [jnp.sqrt(dt / 2) * e1 @ _L for _L in L]
            + [jnp.sqrt(dt / 2) * _L @ e1 for _L in L]
        )
        channel_2 = NestedKrausChannel(
            KrausChannel([jnp.sqrt(dt**2 / 2) * _L1 for _L1 in L]), KrausChannel(L)
        )
        return KrausMap(channel_1, channel_2)


class MESolveFixedRouchon3Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 3 method.
    """

    @staticmethod
    def build_kraus_map(
        H: QArray, L: Sequence[QArray], dt: float, exact_expm: bool
    ) -> KrausMap:
        LdL = sum([_L.dag() @ _L for _L in L])
        G = -1j * H - 0.5 * LdL
        e1o3 = (dt / 3 * G).expm() if exact_expm else _expm_taylor(dt / 3 * G, 3)
        e2o3 = e1o3 @ e1o3
        e3o3 = e2o3 @ e1o3
        channel_1 = KrausChannel(
            [e3o3]
            + [jnp.sqrt(3 * dt / 4) * e1o3 @ _L @ e2o3 for _L in L]
            + [jnp.sqrt(dt / 4) * e3o3 @ _L for _L in L]
        )
        channel_2 = NestedKrausChannel(
            KrausChannel([jnp.sqrt(dt**2 / 2) * e1o3 @ _L1 for _L1 in L]),
            KrausChannel([e1o3 @ _L2 @ e1o3 for _L2 in L]),
        )
        channel_3 = NestedKrausChannel(
            KrausChannel([jnp.sqrt(dt**3 / 6) * _L1 for _L1 in L]),
            KrausChannel(L),
            KrausChannel(L),
        )
        return KrausMap(channel_1, channel_2, channel_3)


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
        def rouchon_step(t0, t1, y0):  # noqa: ANN202
            rho = y0
            t = (t0 + t1) / 2
            dt = t1 - t0

            L, H = self.L(t), self.H(t)

            # === first order
            kraus_map_1 = MESolveFixedRouchon1Integrator.build_kraus_map(
                H, L, dt, self.method.exact_expm
            )
            rho_1 = (
                cholesky_normalize(kraus_map_1, rho) if self.method.normalize else rho
            )
            rho_1 = kraus_map_1(rho_1)

            # === second order
            kraus_map_2 = MESolveFixedRouchon2Integrator.build_kraus_map(
                H, L, dt, self.method.exact_expm
            )
            rho_2 = (
                cholesky_normalize(kraus_map_2, rho) if self.method.normalize else rho
            )
            rho_2 = kraus_map_2(rho_2)

            return rho_2, 0.5 * (rho_2 - rho_1)

        return AbstractRouchonTerm(rouchon_step)


class MESolveAdaptiveRouchon3Integrator(MESolveAdaptiveRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    adaptive Rouchon 2-3 method.
    """

    @property
    def terms(self) -> dx.AbstractTerm:
        def rouchon_step(t0, t1, y0):  # noqa: ANN202
            rho = y0
            t = (t0 + t1) / 2
            dt = t1 - t0

            L, H = self.L(t), self.H(t)

            # === second order
            kraus_map_2 = MESolveFixedRouchon2Integrator.build_kraus_map(
                H, L, dt, self.method.exact_expm
            )
            rho_2 = (
                cholesky_normalize(kraus_map_2, rho) if self.method.normalize else rho
            )
            rho_2 = kraus_map_2(rho_2)

            # === third order
            kraus_map_3 = MESolveFixedRouchon3Integrator.build_kraus_map(
                H, L, dt, self.method.exact_expm
            )
            rho_3 = (
                cholesky_normalize(kraus_map_3, rho) if self.method.normalize else rho
            )
            rho_3 = kraus_map_3(rho_3)
            return rho_3, 0.5 * (rho_3 - rho_2)

        return AbstractRouchonTerm(rouchon_step)


def mesolve_rouchon2_integrator_constructor(**kwargs):
    """Factory function to create a Rouchon2 integrator."""
    if kwargs['method'].dt is not None:
        return MESolveFixedRouchon2Integrator(
            **kwargs,
            diffrax_solver=RouchonDXSolver(2),
            fixed_step=True,
            result_class=MESolveResult,
        )
    return MESolveAdaptiveRouchon2Integrator(
        **kwargs,
        diffrax_solver=AdaptiveRouchonDXSolver(2),
        fixed_step=False,
        result_class=MESolveResult,
    )


def mesolve_rouchon3_integrator_constructor(**kwargs):
    """Factory function to create a Rouchon3 integrator."""
    if kwargs['method'].dt is not None:
        return MESolveFixedRouchon3Integrator(
            **kwargs,
            diffrax_solver=RouchonDXSolver(3),
            fixed_step=True,
            result_class=MESolveResult,
        )
    return MESolveAdaptiveRouchon3Integrator(
        **kwargs,
        diffrax_solver=AdaptiveRouchonDXSolver(3),
        fixed_step=False,
        result_class=MESolveResult,
    )
