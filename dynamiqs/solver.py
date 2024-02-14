from __future__ import annotations

from typing import ClassVar, Literal

import equinox as eqx
from jaxtyping import Scalar

from .gradient import Autograd, CheckpointAutograd, Gradient


class Solver(eqx.Module):
    SUPPORTED_GRADIENT: ClassVar[tuple[Gradient]] = ()
    DEFAULT_GRADIENT: ClassVar[Gradient]

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


class Propagator(Solver):
    SUPPORTED_GRADIENT = (Autograd,)
    DEFAULT_GRADIENT = Autograd()


class _ODESolver(Solver):
    SUPPORTED_GRADIENT = (Autograd, CheckpointAutograd)
    DEFAULT_GRADIENT = CheckpointAutograd()


class _ODEFixedStep(_ODESolver):
    dt: Scalar


class Euler(_ODEFixedStep):
    pass


class Rouchon1(_ODEFixedStep):
    # normalize: The default scheme is trace-preserving at first order only. This
    # parameter sets the normalisation behaviour:
    # - `None`: The scheme is not normalized.
    # - `'sqrt'`: The Kraus map is renormalized with a matrix square root. Ideal
    #   for stiff problems, recommended for time-independent problems.
    # - `cholesky`: The Kraus map is renormalized at each time step using a Cholesky
    #   decomposition. Ideal for stiff problems, recommended for time-dependent
    #   problems.

    normalize: Literal['sqrt', 'cholesky'] | None = None


class Rouchon2(_ODEFixedStep):
    pass


class _ODEAdaptiveStep(_ODESolver):
    atol: float = 1e-8
    rtol: float = 1e-6
    safety_factor: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 5.0
    max_steps: int = 100_000


class Dopri5(_ODEAdaptiveStep):
    pass
