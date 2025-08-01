from __future__ import annotations

from typing import ClassVar

import equinox as eqx
from jaxtyping import PRNGKeyArray
from optimistix import AbstractRootFinder

from ._utils import tree_str_inline
from .gradient import BackwardCheckpointed, Direct, Forward, Gradient

__all__ = [
    'Dopri5',
    'Dopri8',
    'Euler',
    'EulerJump',
    'EulerMaruyama',
    'Expm',
    'JumpMonteCarlo',
    'DiffusiveMonteCarlo',
    'Kvaerno3',
    'Kvaerno5',
    'Rouchon1',
    'Rouchon2',
    'Rouchon3',
    'Tsit5',
    'Event',
]


_TupleGradient = tuple[type[Gradient], ...]


# === generic methods options
class Method(eqx.Module):
    # should be eqx.AbstractClassVar, but this conflicts with the __future__ imports
    SUPPORTED_GRADIENT: ClassVar[_TupleGradient]

    @classmethod
    def supports_gradient(cls, gradient: Gradient | None) -> bool:
        return isinstance(gradient, cls.SUPPORTED_GRADIENT)

    @classmethod
    def assert_supports_gradient(cls, gradient: Gradient | None) -> None:
        if gradient is not None and not cls.supports_gradient(gradient):
            support_str = ', '.join(f'`{x.__name__}`' for x in cls.SUPPORTED_GRADIENT)
            raise ValueError(
                f'Method `{cls.__name__}` does not support gradient'
                f' `{type(gradient).__name__}` (supported gradient types:'
                f' {support_str}).'
            )

    def __str__(self) -> str:
        return tree_str_inline(self)


# === expm method options
class Expm(Method):
    r"""Explicit matrix exponentiation to compute propagators.

    Explicitly batch-compute the propagators for all time intervals in `tsave`. These
    propagators are then iteratively applied:

    - starting from the initial state for [`dq.sesolve()`][dynamiqs.sesolve] and
      [`dq.mesolve()`][dynamiqs.mesolve], to compute states for all times in `tsave`,
    - starting from the identity matrix for [`dq.sepropagator()`][dynamiqs.sepropagator]
      and [`dq.mepropagator()`][dynamiqs.mepropagator], to compute propagators for all
      times in `tsave`.

    For the Schrödinger equation with constant Hamiltonian $H$, the propagator from
    time $t_0$ to time $t_1$ is an $n\times n$ matrix given by
    $$
        U(t_0, t_1) = \exp(-i (t_1 - t_0) H).
    $$

    For the Lindblad master equation with constant Liouvillian $\mathcal{L}$, the
    problem is vectorized and the propagator from time $t_0$ to time $t_1$ is an
    $n^2\times n^2$ matrix given by
    $$
        \mathcal{U}(t_0, t_1) = \exp((t_1 - t_0)\mathcal{L}).
    $$

    Warning:
        If the Hamiltonian or jump operators are sparse qarrays, they will be silently
        converted to dense qarrays before computing their matrix exponentials.

    Warning:
        This method is not recommended for open systems of large dimension, due to
        the $\mathcal{O}(n^6)$ scaling of computing the Liouvillian exponential.

    Warning:
        This method only supports constant or piecewise constant Hamiltonian and jump
        operators.

    Note-: Supported gradients
        This method supports differentiation with
        [`dq.gradient.Direct`][dynamiqs.gradient.Direct] (default).
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (Direct,)

    # dummy init to have the signature in the documentation
    def __init__(self):
        pass


# === generic ODE/SDE methods options
class _DEMethod(Method):
    pass


class _DEFixedStep(_DEMethod):
    dt: float


class _DEAdaptiveStep(_DEMethod):
    rtol: float = 1e-6
    atol: float = 1e-6
    safety_factor: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 5.0
    max_steps: int = eqx.field(static=True, default=100_000)


# === public methods options
class Euler(_DEFixedStep):
    """Euler method (fixed step size ODE method).

    This method is implemented by the [Diffrax](https://docs.kidger.site/diffrax/)
    library, see [`diffrax.Euler`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Euler).

    Warning:
        This method is not recommended for general use.

    Args:
        dt: Fixed time step.

    Note-: Supported gradients
        This method supports differentiation with
        [`dq.gradient.Direct`][dynamiqs.gradient.Direct],
        [`dq.gradient.BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed]
        (default)
        and [`dq.gradient.Forward`][dynamiqs.gradient.Forward].
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (
        Direct,
        BackwardCheckpointed,
        Forward,
    )

    # dummy init to have the signature in the documentation
    def __init__(self, dt: float):
        super().__init__(dt)


class EulerJump(_DEFixedStep):
    r"""Euler-Jump method (fixed step size SDE method).

    Args:
        dt: Fixed time step.

    Note-: Supported gradients
        This method supports differentiation with
        [`dq.gradient.Direct`][dynamiqs.gradient.Direct] (default).
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (Direct,)

    # todo: fix static dt (similar issue as static tsave in dssesolve)
    dt: float = eqx.field(static=True)

    # dummy init to have the signature in the documentation
    def __init__(self, dt: float):
        super().__init__(dt)


class EulerMaruyama(_DEFixedStep):
    r"""Euler-Maruyama method (fixed step size SDE method).

    For a fixed step size $\dt$, it has weak order of convergence $\dt$ and strong order
    of convergence $\sqrt{\dt}$.

    Args:
        dt: Fixed time step.

    Note-: Supported gradients
        This method supports differentiation with
        [`dq.gradient.Direct`][dynamiqs.gradient.Direct] (default).
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (Direct,)

    # todo: fix static dt (similar issue as static tsave in dssesolve)
    dt: float = eqx.field(static=True)

    # dummy init to have the signature in the documentation
    def __init__(self, dt: float):
        super().__init__(dt)


class Rouchon1(_DEFixedStep):
    r"""First-order Rouchon method (fixed step size ODE/SDE method).

    Args:
        dt: Fixed time step.
        normalize: If True, the scheme is trace-preserving to machine precision, which
            is the recommended option because it is much more stable. Otherwise, it is
            only trace-preserving to the scheme order in $\dt$.
        exact_expm: If True, the scheme uses the exact matrix exponential internally (at
            the cost of losing sparsity), otherwise it uses a Taylor expansion up to
            the scheme order.

    Note-: Supported gradients
        This method supports differentiation with
        [`dq.gradient.Direct`][dynamiqs.gradient.Direct],
        [`dq.gradient.BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed]
        (default)
        and [`dq.gradient.Forward`][dynamiqs.gradient.Forward].
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (
        Direct,
        BackwardCheckpointed,
        Forward,
    )

    # todo: fix static dt (similar issue as static tsave in dssesolve)
    dt: float = eqx.field(static=True)
    normalize: bool = eqx.field(static=True, default=True)
    exact_expm: bool = eqx.field(static=True, default=False)

    # dummy init to have the signature in the documentation
    def __init__(self, dt: float, normalize: bool = True, exact_expm: bool = False):
        super().__init__(dt)
        self.normalize = normalize
        self.exact_expm = exact_expm


class Rouchon2(_DEFixedStep, _DEAdaptiveStep):
    r"""Second-order Rouchon method (fixed or adaptive step size ODE method).

    Note:
        By default the scheme is using adaptive step size. The error is estimated
        using the difference between the first-order and second-order. To use a fixed
        step size second-order method (without evaluating the first-order), set the `dt`
        argument.

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        safety_factor: Safety factor for adaptive step sizing.
        min_factor: Minimum factor for adaptive step sizing.
        max_factor: Maximum factor for adaptive step sizing.
        max_steps: Maximum number of steps.
        dt: Fixed time step, if specified all arguments specific to adaptive step sizing
            are ignored (`rtol`, `atol`, `safety_factor`, `min_factor`, `max_factor`
            and `max_steps`).
        normalize: If True, the scheme is trace-preserving to machine precision, which
            is the recommended option because it is much more stable. Otherwise, it is
            only trace-preserving to the scheme order in the numerical step size.
        exact_expm: If True, the scheme uses the exact matrix exponential internally (at
            the cost of losing sparsity), otherwise it uses a Taylor expansion up to
            the scheme order.

    Note-: Supported gradients
        This method supports differentiation with
        [`dq.gradient.Direct`][dynamiqs.gradient.Direct],
        [`dq.gradient.BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed]
        (default)
        and [`dq.gradient.Forward`][dynamiqs.gradient.Forward].
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (
        Direct,
        BackwardCheckpointed,
        Forward,
    )

    normalize: bool = eqx.field(static=True, default=True)
    exact_expm: bool = eqx.field(static=True, default=False)

    # dummy init to have the signature in the documentation
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 100_000,
        dt: float | None = None,
        normalize: bool = True,
        exact_expm: bool = False,
    ):
        _DEFixedStep.__init__(self, dt)
        _DEAdaptiveStep.__init__(
            self, rtol, atol, safety_factor, min_factor, max_factor, max_steps
        )
        self.normalize = normalize
        self.exact_expm = exact_expm


class Rouchon3(_DEFixedStep, _DEAdaptiveStep):
    r"""Third-order Rouchon method (fixed or adaptive step size ODE method).

    Note:
        By default the scheme is using adaptive step size. The error is estimated
        using the difference between the second-order and third-order. To use a fixed
        step size third-order method (without evaluating the second-order), set the `dt`
        argument.

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        safety_factor: Safety factor for adaptive step sizing.
        min_factor: Minimum factor for adaptive step sizing.
        max_factor: Maximum factor for adaptive step sizing.
        max_steps: Maximum number of steps.
        dt: Fixed time step, if specified all arguments specific to adaptive step sizing
            are ignored (`rtol`, `atol`, `safety_factor`, `min_factor`, `max_factor`
            and `max_steps`).
        normalize: If True, the scheme is trace-preserving to machine precision, which
            is the recommended option because it is much more stable. Otherwise, it is
            only trace-preserving to the scheme order in the numerical step size.
        exact_expm: If True, the scheme uses the exact matrix exponential internally (at
            the cost of losing sparsity), otherwise it uses a Taylor expansion up to
            the scheme order.

    Note-: Supported gradients
        This method supports differentiation with
        [`dq.gradient.Direct`][dynamiqs.gradient.Direct],
        [`dq.gradient.BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed]
        (default)
        and [`dq.gradient.Forward`][dynamiqs.gradient.Forward].
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (
        Direct,
        BackwardCheckpointed,
        Forward,
    )

    normalize: bool = eqx.field(static=True, default=True)
    exact_expm: bool = eqx.field(static=True, default=False)

    # dummy init to have the signature in the documentation
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 100_000,
        dt: float | None = None,
        normalize: bool = True,
        exact_expm: bool = False,
    ):
        _DEFixedStep.__init__(self, dt)
        _DEAdaptiveStep.__init__(
            self, rtol, atol, safety_factor, min_factor, max_factor, max_steps
        )
        self.normalize = normalize
        self.exact_expm = exact_expm


class Dopri5(_DEAdaptiveStep):
    """Dormand-Prince method of order 5 (adaptive step size ODE method).

    This method is implemented by the [Diffrax](https://docs.kidger.site/diffrax/)
    library, see [`diffrax.Dopri5`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Dopri5).

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        safety_factor: Safety factor for adaptive step sizing.
        min_factor: Minimum factor for adaptive step sizing.
        max_factor: Maximum factor for adaptive step sizing.
        max_steps: Maximum number of steps.

    Note-: Supported gradients
        This method supports differentiation with
        [`dq.gradient.Direct`][dynamiqs.gradient.Direct],
        [`dq.gradient.BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed]
        (default)
        and [`dq.gradient.Forward`][dynamiqs.gradient.Forward].
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (
        Direct,
        BackwardCheckpointed,
        Forward,
    )

    # dummy init to have the signature in the documentation
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 100_000,
    ):
        super().__init__(rtol, atol, safety_factor, min_factor, max_factor, max_steps)


class Dopri8(_DEAdaptiveStep):
    """Dormand-Prince method of order 8 (adaptive step size ODE method).

    This method is implemented by the [Diffrax](https://docs.kidger.site/diffrax/)
    library, see [`diffrax.Dopri8`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Dopri8).

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        safety_factor: Safety factor for adaptive step sizing.
        min_factor: Minimum factor for adaptive step sizing.
        max_factor: Maximum factor for adaptive step sizing.
        max_steps: Maximum number of steps.

    Note-: Supported gradients
        This method supports differentiation with
        [`dq.gradient.Direct`][dynamiqs.gradient.Direct],
        [`dq.gradient.BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed]
        (default)
        and [`dq.gradient.Forward`][dynamiqs.gradient.Forward].
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (
        Direct,
        BackwardCheckpointed,
        Forward,
    )

    # dummy init to have the signature in the documentation
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 100_000,
    ):
        super().__init__(rtol, atol, safety_factor, min_factor, max_factor, max_steps)


class Tsit5(_DEAdaptiveStep):
    """Tsitouras method of order 5 (adaptive step size ODE method).

    This method is implemented by the [Diffrax](https://docs.kidger.site/diffrax/)
    library, see [`diffrax.Tsit5`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Tsit5).

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        safety_factor: Safety factor for adaptive step sizing.
        min_factor: Minimum factor for adaptive step sizing.
        max_factor: Maximum factor for adaptive step sizing.
        max_steps: Maximum number of steps.

    Note-: Supported gradients
        This method supports differentiation with
        [`dq.gradient.Direct`][dynamiqs.gradient.Direct],
        [`dq.gradient.BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed]
        (default)
        and [`dq.gradient.Forward`][dynamiqs.gradient.Forward].
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (
        Direct,
        BackwardCheckpointed,
        Forward,
    )

    # dummy init to have the signature in the documentation
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 100_000,
    ):
        super().__init__(rtol, atol, safety_factor, min_factor, max_factor, max_steps)


class Kvaerno3(_DEAdaptiveStep):
    """Kvaerno's method of order 3 (adaptive step size and implicit ODE method).

    This method is suitable for stiff problems, typically those with Hamiltonians or
    Liouvillians that have eigenvalues spanning different orders of magnitudes. This is
    for instance the case with problems involving high-order polynomials of the bosonic
    annihilation and creation operators, in large dimensions.

    This method is implemented by the [Diffrax](https://docs.kidger.site/diffrax/)
    library, see [`diffrax.Kvaerno3`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Kvaerno3).

    Warning:
        If you find that your simulation is slow or that the progress bar gets stuck,
        consider switching to double-precision with
        [`dq.set_precision('double')`][dynamiqs.set_precision]. See more details in
        [The sharp bits 🔪](../../documentation/getting_started/sharp-bits.md) tutorial.

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        safety_factor: Safety factor for adaptive step sizing.
        min_factor: Minimum factor for adaptive step sizing.
        max_factor: Maximum factor for adaptive step sizing.
        max_steps: Maximum number of steps.

    Note-: Supported gradients
        This method supports differentiation with
        [`dq.gradient.Direct`][dynamiqs.gradient.Direct],
        [`dq.gradient.BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed]
        (default)
        and [`dq.gradient.Forward`][dynamiqs.gradient.Forward].
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (
        Direct,
        BackwardCheckpointed,
        Forward,
    )

    # dummy init to have the signature in the documentation
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 100_000,
    ):
        super().__init__(rtol, atol, safety_factor, min_factor, max_factor, max_steps)


class Kvaerno5(_DEAdaptiveStep):
    """Kvaerno's method of order 5 (adaptive step size and implicit ODE method).

    This method is suitable for stiff problems, typically those with Hamiltonians or
    Liouvillians that have eigenvalues spanning different orders of magnitudes. This is
    for instance the case with problems involving high-order polynomials of the bosonic
    annihilation and creation operators, in large dimensions.

    This method is implemented by the [Diffrax](https://docs.kidger.site/diffrax/)
    library, see [`diffrax.Kvaerno5`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Kvaerno5).

    Warning:
        If you find that your simulation is slow or that the progress bar gets stuck,
        consider switching to double-precision with
        [`dq.set_precision('double')`][dynamiqs.set_precision]. See more details in
        [The sharp bits 🔪](../../documentation/getting_started/sharp-bits.md) tutorial.

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        safety_factor: Safety factor for adaptive step sizing.
        min_factor: Minimum factor for adaptive step sizing.
        max_factor: Maximum factor for adaptive step sizing.
        max_steps: Maximum number of steps.

    Note-: Supported gradients
        This method supports differentiation with
        [`dq.gradient.Direct`][dynamiqs.gradient.Direct],
        [`dq.gradient.BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed]
        (default)
        and [`dq.gradient.Forward`][dynamiqs.gradient.Forward].
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (
        Direct,
        BackwardCheckpointed,
        Forward,
    )

    # dummy init to have the signature in the documentation
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 100_000,
    ):
        super().__init__(rtol, atol, safety_factor, min_factor, max_factor, max_steps)


class Event(_DEMethod):
    """Event method for the jump SSE.

    This method uses the [Diffrax](https://docs.kidger.site/diffrax/) library event
    handling to interrupt the no-click integration at the next sampled click time,
    apply the corresponding measurement backaction to the state, and continue the
    no-click integration until the subsequent sampled click time.

    Warning: Click times precision
        When using adaptive step size solvers for the no-click integration, you must
        specify either `dtmax` or `root_finder` to control the precision of the click
        times. Otherwise, the adaptive solver may choose to take very large step sizes,
        which results in imprecise click times. You can either:

        - specify `dtmax` to limit the maximum step size of the no-click evolution,
        - or use the `root_finder` argument to refine the exact click times to a chosen
          precision, see for example the
          [optimistix library Newton root finder](https://docs.kidger.site/optimistix/api/root_find/#optimistix.Newton).

    Args:
        noclick_method: Method for the no-click evolution. Defaults to
            [`dq.method.Tsit5`][dynamiqs.method.Tsit5] (supported:
            [`Tsit5`][dynamiqs.method.Tsit5], [`Dopri5`][dynamiqs.method.Dopri5],
            [`Dopri8`][dynamiqs.method.Dopri8],
            [`Kvaerno3`][dynamiqs.method.Kvaerno3],
            [`Kvaerno5`][dynamiqs.method.Kvaerno5],
            [`Euler`][dynamiqs.method.Euler]).
        dtmax: Maximum step size for the no-click evolution, defaults to `None`.
        root_finder: Root finder to refine the click times, defaults to `None`
            (precision determined by the integration step size).
        smart_sampling: If `True`, the no-click trajectory is sampled only once, and
            `result.states` contains only trajectories with one or more clicks.
            Information about the no-click trajectory are stored in `result.infos`:

            - `result.infos.noclick_states` _(qarray of shape (..., nsave, n, 1))_ -
                No-click trajectory.
            - `result.infos.noclick_prob` _(array of shape (...))_ - Probability of
                the no-click trajectory.
            - `result.infos.noclick_expects` _(array of shape
                (..., len(exp_ops), ntsave) or None)_ - No-click trajectory expectation
                values.

    Note-: Supported gradients
        This method supports differentiation with
        [`dq.gradient.BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed]
        (default).
    """

    noclick_method: Method = Tsit5()
    dtmax: float | None = eqx.field(static=True, default=None)
    root_finder: AbstractRootFinder | None = eqx.field(static=True, default=None)
    smart_sampling: bool = eqx.field(static=True, default=False)

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (
        Direct,
        BackwardCheckpointed,
        Forward,
    )

    # dummy init to have the signature in the documentation
    def __init__(
        self,
        noclick_method: Method = Tsit5(),  # noqa: B008
        dtmax: float | None = None,
        root_finder: AbstractRootFinder | None = None,
        smart_sampling: bool = False,
    ):
        self.noclick_method = noclick_method
        self.dtmax = dtmax
        self.root_finder = root_finder
        self.smart_sampling = smart_sampling

        if self.dtmax is not None and isinstance(self.noclick_method, _DEFixedStep):
            raise ValueError(
                'Setting `dtmax` is not compatible with fixed step size methods to '
                'solve the no-click evolution for the `Event` method.'
            )

        if (
            self.dtmax is None
            and self.root_finder is None
            and isinstance(self.noclick_method, _DEAdaptiveStep)
        ):
            raise ValueError(
                'You have to specify either `dtmax` or `root_finder` for the '
                '`Event` method to control the click times precision when using '
                'adaptive step size solvers for the no-click integration. See the '
                '`Event` method documentation for more details.'
            )


class JumpMonteCarlo(_DEMethod):
    """Jump Monte Carlo method for the Lindblad master equation.

    This method calls [`dq.jssesolve()`][dynamiqs.jssesolve] to compute stochastic
    trajectories of the unit-efficiency jump unraveling of the Lindblad master equation
    (ME). These trajectories are then averaged to obtain an approximation of the ME
    solution.

    Note:
        This method is solely a wrapper around [`dq.jssesolve()`][dynamiqs.jssesolve].
        If you are looking for direct access to individual trajectories, use
        [`dq.jssesolve()`][dynamiqs.jssesolve] instead.

    Args:
        keys _(list of PRNG keys)_: PRNG keys used for the jump SSE solver. See
            [`dq.jssesolve()`][dynamiqs.jssesolve] for more details.
        jsse_method: Method used for the jump SSE solver. See
            [`dq.jssesolve()`][dynamiqs.jssesolve] for more details.
        jsse_nmaxclick: Maximum buffer size for `result.clicktimes`. See
            [`dq.jssesolve()`][dynamiqs.jssesolve] for more details.

    Note-: Supported gradients
        See the documentation of the chosen `jsse_method`.
    """

    keys: PRNGKeyArray
    jsse_method: Method
    jsse_nmaxclick: int = eqx.field(static=True, default=10_000)

    # dummy variable, the proper check of gradient support will be done by `jsse_method`
    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (
        Direct,
        BackwardCheckpointed,
        Forward,
    )

    # dummy init to have the signature in the documentation
    def __init__(
        self, keys: PRNGKeyArray, jsse_method: Method, jsse_nmaxclick: int = 10_000
    ):
        self.keys = keys
        self.jsse_method = jsse_method
        self.jsse_nmaxclick = jsse_nmaxclick


class DiffusiveMonteCarlo(_DEMethod):
    """Diffusive Monte Carlo method for the Lindblad master equation.

    This method calls [`dq.dssesolve()`][dynamiqs.dssesolve] to compute stochastic
    trajectories of the unit-efficiency diffusive unraveling of the Lindblad master
    equation (ME). These trajectories are then averaged to obtain an approximation of
    the ME solution.

    Note:
        This method is solely a wrapper around [`dq.dssesolve()`][dynamiqs.dssesolve].
        If you are looking for direct access to individual trajectories, use
        [`dq.dssesolve()`][dynamiqs.dssesolve] instead.

    Args:
        keys _(list of PRNG keys)_: PRNG keys used for the diffusive SSE solver. See
            [`dq.dssesolve()`][dynamiqs.dssesolve] for more details.
        dsse_method: Method used for the diffusive SSE solver. See
            [`dq.dssesolve()`][dynamiqs.dssesolve] for more details.

    Note-: Supported gradients
        See the documentation of the chosen `dsse_method`.
    """

    keys: PRNGKeyArray
    dsse_method: Method

    # dummy variable, the proper check of gradient support will be done by `dsse_method`
    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (
        Direct,
        BackwardCheckpointed,
        Forward,
    )

    # dummy init to have the signature in the documentation
    def __init__(self, keys: PRNGKeyArray, dsse_method: Method):
        self.keys = keys
        self.dsse_method = dsse_method
