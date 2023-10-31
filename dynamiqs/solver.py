from .gradient import AdjointMethod, AutogradMethod


class Solver:
    def __init__(self):
        self.gradient = None


class Propagator(Solver, AutogradMethod):
    pass


class _ODEFixedStep(Solver, AutogradMethod):
    def __init__(self, *, dt: float):
        super().__init__()
        self.dt = dt


class _ODEAdaptiveStep(Solver, AutogradMethod):
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
        super().__init__()
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps
        self.safety_factor = safety_factor
        self.min_factor = min_factor
        self.max_factor = max_factor


class Dopri5(_ODEAdaptiveStep):
    pass


class Euler(_ODEFixedStep, AdjointMethod, AutogradMethod):
    pass


class Rouchon1(_ODEFixedStep, AdjointMethod, AutogradMethod):
    def __init__(self, *, dt: float, sqrt_normalization: bool = False):
        super().__init__(dt=dt)
        self.sqrt_normalization = sqrt_normalization


class Rouchon2(_ODEFixedStep, AdjointMethod, AutogradMethod):
    pass
