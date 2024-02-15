from __future__ import annotations

from typing import ClassVar, Type

import equinox as eqx
from jaxtyping import Scalar

from .gradient import Autograd, CheckpointAutograd, Gradient


# === generic solvers options
class Solver(eqx.Module):
    SUPPORTED_GRADIENT: ClassVar[tuple[Type[Gradient]]] = ()

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


# === generic ODE solvers options
class _ODESolver(Solver):
    pass


class _ODEFixedStep(_ODESolver):
    dt: Scalar


class _ODEAdaptiveStep(_ODESolver):
    rtol: float = 1e-4
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
    pass


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
    pass


class Dopri8(_DiffraxSolver, _ODEAdaptiveStep):
    pass


class Tsit5(_DiffraxSolver, _ODEAdaptiveStep):
    pass
