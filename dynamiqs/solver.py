from __future__ import annotations

from typing import Literal


class Solver:
    pass


class Propagator(Solver):
    pass


class _ODESolver(Solver):
    pass


class _ODEFixedStep(_ODESolver):
    def __init__(self, *, dt: float):
        self.dt = dt


class _ODEAdaptiveStep(Solver):
    def __init__(
        self,
        *,
        rtol: float = 1e-4,
        atol: float = 1e-6,
        max_steps: int = 100_000,
    ):
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps


class Euler(_ODEFixedStep):
    pass


class BackwardEuler(_ODEFixedStep):
    pass


class Rouchon1(_ODEFixedStep):
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
    pass


class Dopri5(_ODEAdaptiveStep):
    pass
