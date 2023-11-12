from __future__ import annotations

from typing import Literal

from torch import Tensor

from .gradient import Adjoint, Autograd, Gradient


class Solver:
    SUPPORTED_GRADIENT = ()

    def supports_gradient(self, gradient: Gradient) -> bool:
        if len(self.SUPPORTED_GRADIENT) == 0:
            return False
        return isinstance(gradient, self.SUPPORTED_GRADIENT)


class Propagator(Solver):
    SUPPORTED_GRADIENT = (Autograd,)


class _ODEFixedStep(Solver):
    SUPPORTED_GRADIENT = (Autograd,)

    def __init__(self, *, dt: float):
        # convert `dt` in case a tensor was passed instead of a float
        if isinstance(dt, Tensor):
            dt = dt.item()
        self.dt = dt


class _ODEAdaptiveStep(Solver):
    SUPPORTED_GRADIENT = (Autograd, Adjoint)

    def __init__(
        self,
        *,
        atol: float = 1e-8,
        rtol: float = 1e-6,
        max_steps: int = 100_000,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
    ):
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps
        self.safety_factor = safety_factor
        self.min_factor = min_factor
        self.max_factor = max_factor


class Dopri5(_ODEAdaptiveStep):
    pass


class Euler(_ODEFixedStep):
    SUPPORTED_GRADIENT = (Autograd, Adjoint)


class Rouchon1(_ODEFixedStep):
    SUPPORTED_GRADIENT = (Autograd, Adjoint)

    def __init__(
        self, *, dt: float, normalize: Literal['sqrt', 'cholesky'] | None = None
    ):
        # normalize: The default scheme is trace-preserving at first order only. This
        # parameter sets the normalisation behaviour:
        # - `None`: The scheme is not normalized.
        # - `'sqrt'`: The Kraus map is renormalized with a matrix square root. Ideal
        #   for stiff problems, recommended for time-independent problems.
        # - `cholesky`: The Kraus map is renormalized at each time step using a Cholesky
        #   decomposition. Ideal for stiff problems, recommended for time-dependent
        #   problems.

        super().__init__(dt=dt)
        self.normalize = normalize


class Rouchon2(_ODEFixedStep):
    SUPPORTED_GRADIENT = (Autograd, Adjoint)
