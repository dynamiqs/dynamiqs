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
from diffrax import Bosh3, Euler, Midpoint, ODETerm
from diffrax._custom_types import RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation

from ...qarrays.layout import dense
from ...qarrays.qarray import QArray
from ...utils.operators import asqarray, eye_like
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


def order2_nojump_evolution(
    H: Callable[[RealScalarLike], QArray],
    L: Callable[[RealScalarLike], Sequence[QArray]],
    t: float,
    dt: float,
) -> Callable[[float], QArray]:
    """Evaluates the no-jump evolution between t and t+dt
    using the explicit midpoint method.
    """
    G0 = -1j * H(t) - 0.5 * sum([_L.dag() @ _L for _L in L(t)])
    Gmid = -1j * H(t + 0.5 * dt) - 0.5 * sum([_L.dag() @ _L for _L in L(t + 0.5 * dt)])
    U0 = eye_like(G0)
    k1 = U0 + G0 * dt / 2
    return U0 + dt * Gmid @ k1


def order3_nojump_dense_evolution(
    H: Callable[[RealScalarLike], QArray],
    L: Callable[[RealScalarLike], Sequence[QArray]],
    t: float,
    dt: float,
) -> Callable[[float], QArray]:
    """Evaluates the no-jump evolution between t and t+dt
    using Kutta's third order method with dense output.
    """
    G0 = -1j * H(t) - 0.5 * sum([_L.dag() @ _L for _L in L(t)])
    Gmid = -1j * H(t + dt / 2) - 0.5 * sum([_L.dag() @ _L for _L in L(t + dt / 2)])
    G1 = -1j * H(t + dt) - 0.5 * sum([_L.dag() @ _L for _L in L(t + dt)])
    U0 = eye_like(G0)
    k1 = G0
    k2 = Gmid @ (U0 + (dt / 2) * k1)
    k3 = G1 @ (U0 - dt * k1 + 2 * dt * k2)
    U1 = U0 + dt / 6 * (k1 + 4 * k2 + k3)

    def interp(s: float) -> QArray:
        # Quadratic Hermite interpolation: p(theta) = a0 + a1*theta + a2*theta^2
        # Constraints: p(0)=U0, p(1)=U1, p'(0)=dt*f0
        theta = (s - t) / dt
        a0 = U0
        a1 = dt * k1
        a2 = U1 - U0 - dt * k1
        return a0 + theta * a1 + theta**2 * a2

    return interp


def solve_propagator(U1, U2) -> QArray:
    """Compute the no jump propagator from t2 to t1 using LU factorization.

    U1: propagator from 0 to t1
    U2: propagator from 0 to t2
    Returns: propagator from t2 to t1
    """
    # U1 = U(t2->t1) @ U2, so U(t2->t1) = U1 @ U2^{-1}
    # Compute U2^{-1} using LU factorization
    return asqarray(jnp.linalg.solve(U2.to_jax().T, U1.to_jax().T).T, dims=U1.dims)


class MESolveFixedRouchonIntegrator(MESolveDiffraxIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using a
    fixed step Rouchon method.
    """

    @property
    def time_dependent(self) -> bool:
        return self.H.time_dependent or any(L.time_dependent for L in self.Ls)

    @property
    def G(self):
        def G_at_t(t) -> QArray:
            LdL = sum([_L.dag() @ _L for _L in self.L(t)])
            return -1j * self.H(t) - 0.5 * LdL

        return G_at_t

    @property
    def identity(self):
        return eye_like(self.H(0), layout=dense)

    @property
    def no_jump_solver(self):
        return Euler()

    @property
    def terms(self) -> dx.AbstractTerm:
        def rouchon_step(t0, t1, y0):  # noqa: ANN202
            # The Rouchon update for a single loss channel is:
            #   rho_{k+1} = sum_k Mk @ rho_k @ Mk^†
            # See comment of `cholesky_normalize()` for the normalization.

            rho = y0
            dt = t1 - t0
            kraus_map = self._build_kraus_map(t0, dt)

            if self.method.normalize:
                rho = cholesky_normalize(kraus_map, rho)

            # for fixed step size, we return None for the error estimate
            return kraus_map(rho), None

        return AbstractRouchonTerm(rouchon_step)

    @property
    def no_jump_propagator(self):
        def _no_jump_propagator_flow(t, y, *args) -> QArray:
            return self.G(t) @ y

        no_jump_propagator_flow = ODETerm(_no_jump_propagator_flow)

        def _no_jump_propagator(t, dt) -> Callable[[RealScalarLike], QArray]:
            solver = self.no_jump_solver
            solver_state = solver.init(
                no_jump_propagator_flow, t, t + dt, self.identity, None
            )
            _, _, dense_info, solver_state, _ = solver.step(
                no_jump_propagator_flow,
                t0=t,
                t1=t + dt,
                y0=self.identity,
                args=None,
                solver_state=solver_state,
                made_jump=False,
            )
            interpolant = solver.interpolation_cls(t0=t, t1=t + dt, **dense_info)
            return interpolant.evaluate

        return _no_jump_propagator

    def _build_kraus_map(self, t: float, dt: float) -> KrausMap:
        return self.build_kraus_map(
            self.no_jump_propagator(t, dt), self.L, t, dt, self.time_dependent
        )

    @staticmethod
    @abstractmethod
    def build_kraus_map(
        no_jump_propagator: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: RealScalarLike,
        dt: RealScalarLike,
        _time_dependent: bool,
    ) -> KrausMap:
        pass


class MESolveFixedRouchon1Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 1 method.
    """

    @property
    def no_jump_solver(self):
        return Euler()

    @staticmethod
    def build_kraus_map(
        no_jump_propagator: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: RealScalarLike,
        dt: RealScalarLike,
        _time_dependent: bool,
    ) -> KrausMap:
        e1 = no_jump_propagator(t + dt)
        channel = KrausChannel([e1] + [jnp.sqrt(dt) * _L for _L in L(t + dt / 2)])
        return KrausMap(channel)


mesolve_rouchon1_integrator_constructor = lambda **kwargs: (
    MESolveFixedRouchon1Integrator(
        **kwargs, diffrax_solver=RouchonDXSolver(1), fixed_step=True
    )
)


class MESolveFixedRouchon2Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 2 method.
    """

    @property
    def no_jump_solver(self):
        return Midpoint()


    @staticmethod
    def build_kraus_map(
        no_jump_propagator: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: RealScalarLike,
        dt: RealScalarLike,
        _time_dependent: bool,
    ) -> KrausMap:
        e1 = no_jump_propagator(t + dt)
        channel_0 = KrausChannel(
            [e1])
        channel_1a = NestedKrausChannel(KrausChannel([jnp.sqrt(dt/2) * e1]), 
                                        KrausChannel(L(t)))
        channel_1b = NestedKrausChannel(KrausChannel(L(t + dt)),
                                        KrausChannel([jnp.sqrt(dt/2) * e1]))
        channel_2 = NestedKrausChannel(
            KrausChannel([jnp.sqrt(dt**2 / 2) * _L1 for _L1 in L(t + 2 * dt / 3)]),
            KrausChannel(L(t + dt / 3)),
        )
        return KrausMap(channel_0, channel_1a, channel_1b, channel_2)


class MESolveFixedRouchon3Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 3 method.
    """

    @property
    def no_jump_solver(self):
        return Bosh3()

    @staticmethod
    def build_kraus_map(
        no_jump_propagator: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: RealScalarLike,
        dt: RealScalarLike,
        _time_dependent: bool,
    ) -> KrausMap:
        e1o3 = no_jump_propagator(t + dt / 3)
        e2o3 = no_jump_propagator(t + 2 * dt / 3)
        e3o3 = no_jump_propagator(t + dt)
        L0o3 = L(t)
        L1o3 = L(t + 1 / 3 * dt)
        L2o3 = L(t + 2 / 3 * dt)
        L1o4 = L(t + dt / 4)
        L2o4 = L(t + dt / 2)
        L3o4 = L(t + 3 * dt / 4)

        # Propagators between the intermediate steps
        e2o3_to_e3o3 = solve_propagator(e3o3, e2o3) if _time_dependent else e1o3
        e1o3_to_e2o3 = solve_propagator(e2o3, e1o3) if _time_dependent else e1o3

        channel_0 = KrausChannel(
            [e3o3])
        channel_1a = NestedKrausChannel(
            KrausChannel([jnp.sqrt(3 * dt / 4) * e2o3_to_e3o3]),
            KrausChannel(L2o3),
            KrausChannel([e2o3 for _L in L2o3])
        )
        channel_1b = NestedKrausChannel(
            KrausChannel([jnp.sqrt(dt / 4) * e3o3]),
            KrausChannel(L0o3)
        )
        channel_2 = NestedKrausChannel(
            KrausChannel([jnp.sqrt(dt**2 / 2) * e2o3_to_e3o3]),
            KrausChannel(L2o3),
            KrausChannel([e1o3_to_e2o3]),
            KrausChannel(L1o3),
            KrausChannel([e1o3]),
        )
        channel_3 = NestedKrausChannel(
            KrausChannel([jnp.sqrt(dt**3 / 6) * _L1 for _L1 in L3o4]),
            KrausChannel(L2o4),
            KrausChannel(L1o4),
        )
        return KrausMap(channel_0, channel_1a, channel_1b, channel_2, channel_3)


class MESolveAdaptiveRouchonIntegrator(MESolveDiffraxIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using an
    adaptive Rouchon method.
    """

    @property
    def G(self):
        def G_at_t(t) -> QArray:
            LdL = sum([_L.dag() @ _L for _L in self.L(t)])
            return -1j * self.H(t) - 0.5 * LdL

        return G_at_t

    @property
    def identity(self):
        return eye_like(self.H(0), layout=dense)

    @property
    def no_jump_solver_low(self):
        pass

    @property
    def no_jump_solver_high(self):
        pass

    @property
    def time_dependent(self) -> bool:
        return self.H.time_dependent or any(L.time_dependent for L in self.Ls)

    @property
    def no_jump_propagators(self):
        no_jump_propagator_term = ODETerm(lambda t, y, _args: self.G(t) @ y)

        def _no_jump_propagator_low(t, dt) -> Callable[[RealScalarLike], QArray]:
            solver_low = self.no_jump_solver_low
            state_low = solver_low.init(
                no_jump_propagator_term, t, t + dt, self.identity, None
            )
            _, _, dense_info, state_low, _ = solver_low.step(
                no_jump_propagator_term,
                t0=t,
                t1=t + dt,
                y0=self.identity,
                args=None,
                solver_state=state_low,
                made_jump=False,
            )
            interpolant_low = solver_low.interpolation_cls(
                t0=t, t1=t + dt, **dense_info
            )
            return interpolant_low.evaluate

        def _no_jump_propagator_high(t, dt) -> Callable[[RealScalarLike], QArray]:
            solver_high = self.no_jump_solver_high
            state_high = solver_high.init(
                no_jump_propagator_term, t, t + dt, self.identity, None
            )
            _, _, dense_info, state_high, _ = solver_high.step(
                no_jump_propagator_term,
                t0=t,
                t1=t + dt,
                y0=self.identity,
                args=None,
                solver_state=state_high,
                made_jump=False,
            )
            interpolant_high = solver_high.interpolation_cls(
                t0=t, t1=t + dt, **dense_info
            )
            return interpolant_high.evaluate

        return _no_jump_propagator_low, _no_jump_propagator_high

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
    def no_jump_solver_low(self):
        return Euler()

    @property
    def no_jump_solver_high(self):
        return Midpoint()

    @property
    def terms(self) -> dx.AbstractTerm:
        def rouchon_step(t0, t1, y0):  # noqa: ANN202
            rho = y0
            dt = t1 - t0
            no_jump_propagator_low, no_jump_propagator_high = [
                no_jump_propagator(t0, dt)
                for no_jump_propagator in self.no_jump_propagators
            ]
            # === first order
            kraus_map_1 = MESolveFixedRouchon1Integrator.build_kraus_map(
                no_jump_propagator_low, self.L, t0, dt, self.time_dependent
            )
            rho_1 = (
                cholesky_normalize(kraus_map_1, rho) if self.method.normalize else rho
            )
            rho_1 = kraus_map_1(rho_1)

            # === second order
            kraus_map_2 = MESolveFixedRouchon2Integrator.build_kraus_map(
                no_jump_propagator_high, self.L, t0, dt, self.time_dependent
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
    def no_jump_solver_low(self):
        return Midpoint()

    @property
    def no_jump_solver_high(self):
        return Bosh3()

    @property
    def terms(self) -> dx.AbstractTerm:
        def rouchon_step(t0, t1, y0):  # noqa: ANN202
            rho = y0
            dt = t1 - t0

            no_jump_propagator_low, no_jump_propagator_high = [
                no_jump_propagator(t0, dt)
                for no_jump_propagator in self.no_jump_propagators
            ]

            # === second order
            kraus_map_2 = MESolveFixedRouchon2Integrator.build_kraus_map(
                no_jump_propagator_low, self.L, t0, dt, self.time_dependent
            )
            rho_2 = (
                cholesky_normalize(kraus_map_2, rho) if self.method.normalize else rho
            )
            rho_2 = kraus_map_2(rho_2)

            # === third order
            kraus_map_3 = MESolveFixedRouchon3Integrator.build_kraus_map(
                no_jump_propagator_high, self.L, t0, dt, self.time_dependent
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
