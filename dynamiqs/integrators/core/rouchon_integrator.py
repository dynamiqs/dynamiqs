from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import replace

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import AbstractRungeKutta, Bosh3, Euler, Midpoint, ODETerm
from diffrax._custom_types import RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation

from ...gradient import Forward
from ...qarrays.layout import dense
from ...qarrays.qarray import QArray
from ...utils.operators import eye_like
from .diffrax_integrator import MESolveDiffraxIntegrator


def M_rho_Mdag(M: QArray, rho: QArray) -> QArray:
    return M @ rho @ M.dag()


def Mdag_O_M(M: QArray, O: QArray) -> QArray:
    return M.dag() @ O @ M


def Mdag_M(M: QArray) -> QArray:
    return M.dag() @ M


class AbstractRouchonTerm(dx.AbstractTerm):
    # this class bypasses the typical Diffrax term implementation, as Rouchon schemes
    # don't match the vf/contr/prod structure

    rouchon_step: Callable[[RealScalarLike, RealScalarLike, Y], [Y, Y]]
    # should be defined as `rouchon_step(t0, t1, y0) -> y1, error`

    def vf(self, t: RealScalarLike, y: Y, args: object):
        del t, y, args

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs: object):
        del t0, t1, kwargs

    def prod(self, vf: object, control: object):
        del vf, control


class RouchonDXSolver(dx.AbstractSolver):
    _order: int
    term_structure = AbstractRouchonTerm
    interpolation_cls = LocalLinearInterpolation

    def init(
        self,
        terms: AbstractRouchonTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: object,
    ):
        del terms, t0, t1, y0, args

    def step(
        self,
        terms: AbstractRouchonTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: object,
        solver_state: None,
        made_jump: bool,
    ) -> tuple:
        del solver_state, made_jump, args
        y1, error = terms.term.rouchon_step(t0, t1, y0)
        dense_info = dict(y0=y0, y1=y1)
        return y1, error, dense_info, None, dx.RESULTS.successful

    def func(self, terms: AbstractRouchonTerm, t0: RealScalarLike, y0: Y, args: object):
        del terms, t0, y0, args

    def order(self, terms: AbstractRouchonTerm) -> int:
        del terms
        return self._order


class AdaptiveRouchonDXSolver(dx.AbstractAdaptiveSolver, RouchonDXSolver):
    pass


class KrausRK(eqx.Module):
    r"""Kraus-form Runge-Kutta method for the Lindblad master equation.

    A Rouchon RK method is defined by a Butcher tableau $(c, A, b)$:

    - $c$: stage nodes (length $s$)
    - $A$: stage coefficient matrix ($s \times s$, lower triangular for explicit
      methods)
    - $b$: output weights (length $s$)

    The scheme computes intermediate stages:
    $$\rho^{(i)} = U(c_i) \rho_0 U(c_i)^\dagger
        + \Delta t \sum_j a_{ij} P_{ij} D_j(\rho^{(j)}) P_{ij}^\dagger$$
    where $D_j(\rho) = \sum_k L_k(c_j) \rho L_k(c_j)^\dagger$ and
    $P_{ij} = U(c_i) \, U(c_j)^{-1}$ is the propagator ratio.

    The output is:
    $$\rho_\text{new} = U(1.0) \rho_0 U(1.0)^\dagger
        + \Delta t \sum_i b_i \, Q_i \, D_i(\rho^{(i)}) \, Q_i^\dagger$$
    where $Q_i = U(1.0) \, U(c_i)^{-1}$.

    Subclasses define the Butcher tableau via class attributes ``_c``, ``_A``,
    ``_b``, and ``_neumann_order`` (order of Neumann series for propagator
    inverses).

    Attributes:
        nojump_propagator: Interpolant mapping an absolute time to the no-jump
            propagator from ``t`` to that time.
        t: Beginning of the time step.
        dt: Time step.
        Ls: Function mapping an absolute time to the jump operators at that time.
        identity: Identity operator.
    """

    nojump_propagator: Callable[[RealScalarLike], QArray]
    t: RealScalarLike
    dt: RealScalarLike
    Ls: Callable[[RealScalarLike], Sequence[QArray]]
    identity: QArray

    # Butcher tableau — overridden by subclasses (not pytree leaves)
    _c = ()
    _A = ()
    _b = ()
    _neumann_order = 0

    # -- propagator helpers --------------------------------------------------

    def U(self, c: float) -> QArray:
        r"""No-jump propagator from 0 to $c \Delta t$ (relative to $t$)."""
        if c == 0.0:
            return self.identity
        return self.nojump_propagator(self.t + c * self.dt)

    def propagator_ratio(self, ci: float, cj: float) -> QArray:
        r"""Propagator ratio $U(c_i) U(c_j)^{-1}$ via Neumann approximation."""
        if ci == cj:
            return self.identity
        U_ci = self.U(ci)
        if cj == 0.0:
            return U_ci
        # Neumann series: U^{-1} ≈ I + W + W² + ... with W = I - U
        U_cj = self.U(cj)
        W = self.identity - U_cj
        inv_approx = self.identity
        W_power = self.identity
        for _ in range(self._neumann_order):
            W_power = W_power @ W
            inv_approx = inv_approx + W_power
        return U_ci @ inv_approx

    # -- forward pass --------------------------------------------------------

    def dissipator(self, c: float, rho: QArray) -> QArray:
        r"""Jump map $D_c(\rho) = \sum_k L_k \rho L_k^\dagger$ at $t + c\Delta t$."""
        return sum(M_rho_Mdag(L, rho) for L in self.Ls(self.t + c * self.dt))

    def compute_stages(self, rho0: QArray) -> list[QArray]:
        r"""Compute all intermediate stages $\rho^{(0)}, \ldots, \rho^{(s-1)}$."""
        stages: list[QArray] = []
        for i, ci in enumerate(self._c):
            rho_i = M_rho_Mdag(self.U(ci), rho0)
            for j in range(i):
                a_ij = self._A[i][j]
                if a_ij == 0.0:
                    continue
                cj = self._c[j]
                P_ij = self.propagator_ratio(ci, cj)
                rho_i = rho_i + self.dt * a_ij * M_rho_Mdag(
                    P_ij, self.dissipator(cj, stages[j])
                )
            stages.append(rho_i)
        return stages

    def __call__(self, rho0: QArray) -> QArray:
        stages = self.compute_stages(rho0)
        rho_new = M_rho_Mdag(self.U(1.0), rho0)
        for i, bi in enumerate(self._b):
            if bi == 0.0:
                continue
            ci = self._c[i]
            Q_i = self.propagator_ratio(1.0, ci)
            rho_new = rho_new + self.dt * bi * M_rho_Mdag(
                Q_i, self.dissipator(ci, stages[i])
            )
        return rho_new

    # -- adjoint pass (for normalization) ------------------------------------

    def adjoint_dissipator(self, c: float, O: QArray) -> QArray:
        r"""Adjoint jump map $D_c^*(O) = \sum_k L_k^\dagger O L_k$."""
        return sum(Mdag_O_M(L, O) for L in self.Ls(self.t + c * self.dt))

    def S_stage(self, i: int, O: QArray) -> QArray:
        r"""Backward propagation of observation $O$ through stage $i$.

        Satisfies $\mathrm{tr}(O\,\rho^{(i)}) = \mathrm{tr}(S_i(O)\,\rho_0)$.
        """
        ci = self._c[i]
        result = Mdag_O_M(self.U(ci), O)
        for j in range(i):
            a_ij = self._A[i][j]
            if a_ij == 0.0:
                continue
            cj = self._c[j]
            P_ij = self.propagator_ratio(ci, cj)
            jump_obs = self.dt * a_ij * self.adjoint_dissipator(cj, Mdag_O_M(P_ij, O))
            result = result + self.S_stage(j, jump_obs)
        return result

    def S(self) -> QArray:
        r"""Compute $S = \sum_k M_k^\dagger M_k$, used for Cholesky normalization."""
        S = Mdag_M(self.U(1.0))
        for i, bi in enumerate(self._b):
            if bi == 0.0:
                continue
            ci = self._c[i]
            Q_i = self.propagator_ratio(1.0, ci)
            obs = self.adjoint_dissipator(ci, Mdag_M(Q_i))
            S = S + self.dt * bi * self.S_stage(i, obs)
        return S


class KrausEuler(KrausRK):
    r"""First-order Rouchon method (Euler in Kraus form).

    Butcher tableau::

        0 | 0
        -----
          | 1
    """

    _c = (0.0,)
    _A = ((0.0,),)
    _b = (1.0,)
    _neumann_order = 0

    def __call__(self, rho0: QArray) -> QArray:
        # override to avoid computing the single stage as an intermediate stage
        return M_rho_Mdag(self.U(1.0), rho0) + self.dt * self.dissipator(0.0, rho0)

    def S(self) -> QArray:
        # override to avoid computing the single stage as an intermediate stage
        nojump_S = Mdag_M(self.U(1.0))
        jump_S = self.dt * sum(Mdag_M(_L) for _L in self.Ls(self.t))
        return nojump_S + jump_S

    def get_kraus_operators(self) -> list[QArray]:
        # this method is only used for stochastic solvers
        nojump_kraus_ops = [self.U(1.0)]
        jump_kraus_ops = [jnp.sqrt(self.dt) * _L for _L in self.Ls(self.t)]
        return nojump_kraus_ops + jump_kraus_ops


class KrausHeun2(KrausRK):
    r"""Second-order Rouchon method based on Heun's method.

    Butcher tableau::

        0   |
        1   | 1
        ---------
            | 1/2  1/2
    """

    _c = (0.0, 1.0)
    _A = ((0.0, 0.0), (1.0, 0.0))
    _b = (0.5, 0.5)
    _neumann_order = 0


class KrausHeun3(KrausRK):
    r"""Third-order Rouchon method based on Heun's third-order method.

    Butcher tableau::

        0     |
        1/3   | 1/3
        2/3   | 0    2/3
        -------------------
              | 1/4  0    3/4
    """

    _c = (0.0, 1 / 3, 2 / 3)
    _A = ((0.0, 0.0, 0.0), (1 / 3, 0.0, 0.0), (0.0, 2 / 3, 0.0))
    _b = (0.25, 0.0, 0.75)
    _neumann_order = 2


def cholesky_normalize(kraus_map: KrausRK, rho: QArray) -> jax.Array:
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
        T, rho, lower=True, transpose_a=True, conjugate_a=True, left_side=True
    )
    # solve x @ T = rho => x = rho @ T^{-1}
    return jax.lax.linalg.triangular_solve(T, rho, lower=True, left_side=False)


class RouchonPropertiesMixin:
    """Mixin providing shared properties for Rouchon integrators.

    Subclasses must define ``nojump_diffrax_solver``.
    Expects ``self.H`` and ``self.L`` to be callable.
    """

    def G(self, t: RealScalarLike) -> QArray:
        LdL = sum(Mdag_M(_L) for _L in self.L(t))
        return -1j * self.H(t) - 0.5 * LdL

    @property
    def identity(self) -> QArray:
        return eye_like(self.H(0), layout=dense)

    @property
    @abstractmethod
    def nojump_diffrax_solver(self) -> dx.AbstractSolver:
        pass

    def make_nojump_propagator(
        self, t: RealScalarLike, dt: RealScalarLike
    ) -> Callable[[RealScalarLike], QArray]:
        """Compute interpolant for the no-jump propagator over [t, t+dt].

        Returns a function mapping an absolute time in [t, t+dt] to the
        no-jump propagator at that time. Uses dense output so that higher-order
        Rouchon schemes (e.g. RK3) can evaluate intermediate propagators.
        """

        def _nojump_propagator_flow(t, y, *args) -> QArray:  # noqa: ANN001
            return self.G(t) @ y

        term = ODETerm(_nojump_propagator_flow)
        solver = self._make_forward_compatible(self.nojump_diffrax_solver)
        t1 = t + dt
        y0 = self.identity
        solver_state = solver.init(term, t, t1, y0, None)
        _, _, dense_info, _, _ = solver.step(
            term,
            t0=t,
            t1=t1,
            y0=y0,
            args=None,
            solver_state=solver_state,
            made_jump=False,
        )
        interpolant = solver.interpolation_cls(t0=t, t1=t1, **dense_info)
        return interpolant.evaluate

    def _make_forward_compatible(self, solver: dx.AbstractSolver) -> dx.AbstractSolver:
        """Make the solver forward-mode compatible.

        This is the same hack used in diffrax to make `AbstractRungeKutta` solvers
        forward-mode compatible, see https://github.com/patrick-kidger/diffrax/blob/5ea5fcb05058bd22548300a10ea5ef5ce7fb75bb/diffrax/_adjoint.py#L381-L386.
        """
        if (
            isinstance(self.gradient, Forward)
            and isinstance(solver, AbstractRungeKutta)
            and solver.scan_kind is None
        ):
            return eqx.tree_at(
                lambda s: s.scan_kind, solver, 'bounded', is_leaf=lambda x: x is None
            )
        return solver


class MESolveFixedRouchonIntegrator(RouchonPropertiesMixin, MESolveDiffraxIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using a
    fixed step Rouchon method.

    Subclasses must set ``_kraus_cls`` and may override ``nojump_diffrax_solver``.
    """

    @property
    def terms(self) -> dx.AbstractTerm:
        def rouchon_step(t0, t1, y0):  # noqa: ANN001, ANN202
            # The Rouchon update for a single loss channel is:
            #   rho_{k+1} = sum_k Mk @ rho_k @ Mk^†
            # See comment of `cholesky_normalize()` for the normalization.

            rho = y0
            dt = t1 - t0
            nojump_propagator = self.make_nojump_propagator(t0, dt)
            kraus_map = self.build_kraus_map(
                nojump_propagator, self.L, t0, dt, self.identity
            )

            if self.method.normalize:
                rho = cholesky_normalize(kraus_map, rho)

            return kraus_map(rho), None

        return AbstractRouchonTerm(rouchon_step)

    @classmethod
    def build_kraus_map(
        cls,
        nojump_propagator: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: RealScalarLike,
        dt: RealScalarLike,
        identity: QArray,
    ) -> KrausRK:
        return cls._kraus_cls(
            nojump_propagator=nojump_propagator, t=t, dt=dt, Ls=L, identity=identity
        )


class MESolveFixedRouchon1Integrator(MESolveFixedRouchonIntegrator):
    """Fixed step Rouchon 1 (Euler) integrator for the Lindblad master equation."""

    _kraus_cls = KrausEuler

    @property
    def nojump_diffrax_solver(self) -> dx.AbstractSolver:
        return Euler()


class MESolveFixedRouchon2Integrator(MESolveFixedRouchonIntegrator):
    """Fixed step Rouchon 2 (Heun) integrator for the Lindblad master equation."""

    _kraus_cls = KrausHeun2

    @property
    def nojump_diffrax_solver(self) -> dx.AbstractSolver:
        return Midpoint()


class MESolveFixedRouchon3Integrator(MESolveFixedRouchonIntegrator):
    """Fixed step Rouchon 3 (Heun3) integrator for the Lindblad master equation."""

    _kraus_cls = KrausHeun3

    @property
    def nojump_diffrax_solver(self) -> dx.AbstractSolver:
        return Bosh3()


class MESolveAdaptiveRouchonIntegrator(
    RouchonPropertiesMixin, MESolveDiffraxIntegrator
):
    """Integrator computing the time evolution of the Lindblad master equation using an
    adaptive Rouchon method.

    Subclasses must set ``_solver_low``, ``_solver_high``, ``_fixed_cls_low``,
    and ``_fixed_cls_high`` as class attributes, and may override
    ``_build_dense_info_low`` for solvers that need extra interpolation data.
    """

    def _build_dense_info_low(
        self,
        y0: QArray,
        y1_low: QArray,
        dense_info_high: dict,  # noqa: ARG002
    ) -> dict:
        """Build dense_info for the low-order interpolant. Override for extra keys."""
        return dict(y0=y0, y1=y1_low)

    def make_nojump_propagators(
        self, t: RealScalarLike, dt: RealScalarLike
    ) -> tuple[Callable[[RealScalarLike], QArray], Callable[[RealScalarLike], QArray]]:
        """Compute embedded low/high-order no-jump propagator interpolants.

        Uses the high-order solver's error estimate to derive both a low-order
        and high-order propagator from a single step.
        """

        def _nojump_propagator_flow(t, y, *args) -> QArray:  # noqa: ANN001
            return self.G(t) @ y

        term = ODETerm(_nojump_propagator_flow)
        solver_low = self._make_forward_compatible(self._solver_low)
        solver_high = self._make_forward_compatible(self._solver_high)
        t1 = t + dt
        y0 = self.identity

        solver_state = solver_high.init(term, t, t1, y0, None)
        y1_high, error, dense_info_high, solver_state, _result = solver_high.step(
            term,
            t0=t,
            t1=t1,
            y0=y0,
            args=None,
            solver_state=solver_state,
            made_jump=False,
        )

        y1_low = y1_high - error
        dense_info_low = self._build_dense_info_low(y0, y1_low, dense_info_high)

        interpolant_low = solver_low.interpolation_cls(t0=t, t1=t1, **dense_info_low)
        interpolant_high = solver_high.interpolation_cls(t0=t, t1=t1, **dense_info_high)

        return interpolant_low.evaluate, interpolant_high.evaluate

    @property
    def terms(self) -> dx.AbstractTerm:
        def rouchon_step(t0, t1, y0):  # noqa: ANN001, ANN202
            rho = y0
            dt = t1 - t0
            propagator_low, propagator_high = self.make_nojump_propagators(t0, dt)

            # low order
            kraus_low = self._fixed_cls_low.build_kraus_map(
                propagator_low, self.L, t0, dt, self.identity
            )
            rho_low = (
                cholesky_normalize(kraus_low, rho) if self.method.normalize else rho
            )
            rho_low = kraus_low(rho_low)

            # high order
            kraus_high = self._fixed_cls_high.build_kraus_map(
                propagator_high, self.L, t0, dt, self.identity
            )
            rho_high = (
                cholesky_normalize(kraus_high, rho) if self.method.normalize else rho
            )
            rho_high = kraus_high(rho_high)

            return rho_high, 0.5 * (rho_high - rho_low)

        return AbstractRouchonTerm(rouchon_step)

    @property
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        # todo: can we do better?
        stepsize_controller = super().stepsize_controller
        # fix incorrect default linear interpolation by stepping exactly at all times
        # in tsave, so interpolation is bypassed
        return replace(stepsize_controller, step_ts=self.ts)


class MESolveAdaptiveRouchon2Integrator(MESolveAdaptiveRouchonIntegrator):
    """Adaptive Rouchon 1-2 integrator with embedded dense outputs from Midpoint."""

    _solver_low = Euler()
    _solver_high = Midpoint()
    _fixed_cls_low = MESolveFixedRouchon1Integrator
    _fixed_cls_high = MESolveFixedRouchon2Integrator


class MESolveAdaptiveRouchon3Integrator(MESolveAdaptiveRouchonIntegrator):
    """Adaptive Rouchon 2-3 integrator with embedded dense outputs from Bosh3."""

    _solver_low = Midpoint()
    _solver_high = Bosh3()
    _fixed_cls_low = MESolveFixedRouchon2Integrator
    _fixed_cls_high = MESolveFixedRouchon3Integrator

    def _build_dense_info_low(
        self, y0: QArray, y1_low: QArray, dense_info_high: dict
    ) -> dict:
        # Midpoint interpolation needs the k stages from Bosh3
        k = dense_info_high['k']
        return dict(y0=y0, y1=y1_low, k=k[:2])


mesolve_rouchon1_integrator_constructor = lambda **kwargs: (
    MESolveFixedRouchon1Integrator(
        **kwargs, diffrax_solver=RouchonDXSolver(1), fixed_step=True
    )
)


def mesolve_rouchon2_integrator_constructor(**kwargs) -> MESolveDiffraxIntegrator:
    """Factory function to create a Rouchon2 integrator."""
    if kwargs['method'].dt is not None:
        return MESolveFixedRouchon2Integrator(
            **kwargs, diffrax_solver=RouchonDXSolver(2), fixed_step=True
        )
    return MESolveAdaptiveRouchon2Integrator(
        **kwargs, diffrax_solver=AdaptiveRouchonDXSolver(2), fixed_step=False
    )


def mesolve_rouchon3_integrator_constructor(**kwargs) -> MESolveDiffraxIntegrator:
    """Factory function to create a Rouchon3 integrator."""
    if kwargs['method'].dt is not None:
        return MESolveFixedRouchon3Integrator(
            **kwargs, diffrax_solver=RouchonDXSolver(3), fixed_step=True
        )
    return MESolveAdaptiveRouchon3Integrator(
        **kwargs, diffrax_solver=AdaptiveRouchonDXSolver(3), fixed_step=False
    )
