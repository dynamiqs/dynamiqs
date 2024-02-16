from __future__ import annotations

from typing import ClassVar, Type

import equinox as eqx
from jaxtyping import Scalar

from .gradient import Autograd, CheckpointAutograd, Gradient

__all__ = ['Propagator', 'Euler', 'Rouchon1', 'Rouchon2', 'Dopri5', 'Dopri8', 'Tsit5']

# === generic solvers options
class Solver(eqx.Module):
    SUPPORTED_GRADIENT: ClassVar[tuple[Type[Gradient], ...]] = ()

    @classmethod
    def supports_gradient(cls, gradient: Gradient | None) -> bool:
        return isinstance(gradient, cls.SUPPORTED_GRADIENT)

    @classmethod
    def assert_supports_gradient(cls, gradient: Gradient | None) -> None:
        if gradient is not None and not cls.supports_gradient(gradient):
            support_str = ', '.join(f'`{x.__name__}`' for x in cls.SUPPORTED_GRADIENT)
            raise ValueError(
                f'Solver `{cls.__name__}` does not support gradient'
                f' `{type(gradient).__name__}` (supported gradient types:'
                f' {support_str}).'
            )


# === propagator solvers options
class Propagator(Solver):
    SUPPORTED_GRADIENT = (Autograd,)

    def __init__(self):
        r"""Quantum propagator method.

        Compute the exact quantum propagator from matrix exponentiation. For the Schrödinger equation with Hamiltonian $H$, the propagator is given by
        $$
            U(t_0, t_1) = \exp(-i H (t_1 - t_0)),
        $$
        For the Lindblad master equation with Liouvillian $\mathcal{L}$, the problem is vectorized and the propagator is given by
        $$
            \mathcal{U}(t_0, t_1) = \exp(-i \mathcal{L} (t_1 - t_0)),
        $$

        Note: Constant problem support only.
            The propagator method only supports constant Hamiltonians and jump
            operators for now. Piecewise-constant problems will be supported in the
            future.
        """
        self.assert_supports_gradient(Autograd)


# === generic ODE solvers options
class _ODESolver(Solver):
    pass


class _ODEFixedStep(_ODESolver):
    dt: float


class _ODEAdaptiveStep(_ODESolver):
    rtol: float = 1e-6
    atol: float = 1e-6
    safety_factor: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 5.0
    max_steps: int = 100_000


# === diffrax-based solvers options
class _DiffraxSolver(Solver):
    SUPPORTED_GRADIENT = (Autograd, CheckpointAutograd)


# === public solvers options
class Euler(_DiffraxSolver, _ODEFixedStep):
    def __init__(self, dt: float):
        """Euler's method (from [Diffrax](https://docs.kidger.site/diffrax/)).

        1st order explicit Runge--Kutta method. Does not support adaptive step sizing.
        Uses 1 stage. Uses 1st order local linear interpolation for dense `tsave`
        output.

        Args:
            dt _(float)_: Time step.
        """
        _ODEFixedStep().__init__(dt)

class Rouchon1(_DiffraxSolver, _ODEFixedStep):
    # normalize: The default scheme is trace-preserving at first order only. This
    # parameter sets the normalisation behaviour:
    # - `None`: The scheme is not normalized.
    # - `'sqrt'`: The Kraus map is renormalized with a matrix square root. Ideal
    #   for stiff problems, recommended for time-independent problems.
    # - `cholesky`: The Kraus map is renormalized at each time step using a Cholesky
    #   decomposition. Ideal for stiff problems, recommended for time-dependent
    #   problems.

    # todo: fix, strings are not valid JAX types
    # normalize: Literal['sqrt', 'cholesky'] | None = None
    pass


class Rouchon2(_DiffraxSolver, _ODEFixedStep):
    pass


class Dopri5(_DiffraxSolver, _ODEAdaptiveStep):
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 100_000,
    ):
        """
        Dormand--Prince's 5/4 method (from
        [Diffrax](https://docs.kidger.site/diffrax/)).

        5th order Runge--Kutta method. Has an embedded 4th order method for adaptive step sizing. Uses 7 stages with first same as last. Uses 5th order interpolation for dense `tsave` output.

        Args:
            rtol _(float, optional)_: Relative tolerance. Defaults to 1e-6.
            atol _(float, optional)_: Absolute tolerance. Defaults to 1e-6.
            safety_factor _(float, optional)_: Safety factor for step sizing. Defaults
                to 0.9.
            min_factor _(float, optional)_: Minimum factor for step sizing. Defaults to
                0.2.
            max_factor _(float, optional)_: Maximum factor for step sizing. Defaults to
                5.0.
            max_steps _(int, optional)_: Maximum number of steps. Defaults to 100_000.
        """
        _ODEAdaptiveStep().__init__(rtol, atol, safety_factor, min_factor, max_factor, max_steps)


class Dopri8(_DiffraxSolver, _ODEAdaptiveStep):
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 100_000,
    ):
        """
        Dormand--Prince's 8/7 method (from
        [Diffrax](https://docs.kidger.site/diffrax/)).

        8th order Runge--Kutta method. Has an embedded 7th order method for adaptive
        step sizing. Uses 14 stages with first same as last. Uses 8th order
        interpolation for dense `tsave` output.

        Args:
            rtol _(float, optional)_: Relative tolerance. Defaults to 1e-6.
            atol _(float, optional)_: Absolute tolerance. Defaults to 1e-6.
            safety_factor _(float, optional)_: Safety factor for step sizing. Defaults
                to 0.9.
            min_factor _(float, optional)_: Minimum factor for step sizing. Defaults to
                0.2.
            max_factor _(float, optional)_: Maximum factor for step sizing. Defaults to
                5.0.
            max_steps _(int, optional)_: Maximum number of steps. Defaults to 100_000.
        """
        _ODEAdaptiveStep().__init__(rtol, atol, safety_factor, min_factor, max_factor, max_steps)


class Tsit5(_DiffraxSolver, _ODEAdaptiveStep):
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 100_000,
    ):
        """Tsitouras' 5/4 method (from [Diffrax](https://docs.kidger.site/diffrax/)).

        5th order explicit Runge--Kutta method. Has an embedded 4th order method for
        adaptive step sizing. Uses 7 stages with first same as last. Uses 5th order
        interpolation for dense `tsave` output.

        Args:
            rtol _(float, optional)_: Relative tolerance. Defaults to 1e-6.
            atol _(float, optional)_: Absolute tolerance. Defaults to 1e-6.
            safety_factor _(float, optional)_: Safety factor for step sizing. Defaults
                to 0.9.
            min_factor _(float, optional)_: Minimum factor for step sizing. Defaults to
                0.2.
            max_factor _(float, optional)_: Maximum factor for step sizing. Defaults to
                5.0.
            max_steps _(int, optional)_: Maximum number of steps. Defaults to 100_000.
        """
        _ODEAdaptiveStep().__init__(rtol, atol, safety_factor, min_factor, max_factor, max_steps)
