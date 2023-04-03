class SolverOption:
    def __init__(self, *, verbose: bool = True):
        self.verbose = verbose


class FixedStep(SolverOption):
    def __init__(self, *, dt: float, verbose: bool = True):
        super().__init__(verbose=verbose)
        self.dt = dt


class AdaptiveStep(SolverOption):
    def __init__(
        self,
        *,
        atol: float = 1e-8,
        rtol: float = 1e-6,
        max_steps: int = 100_000,
        factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        verbose: bool = True,
    ):
        super().__init__(verbose=verbose)
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps
        self.factor = factor
        self.min_factor = min_factor
        self.max_factor = max_factor


class Euler(FixedStep):
    pass


class Dopri45(AdaptiveStep):
    pass
