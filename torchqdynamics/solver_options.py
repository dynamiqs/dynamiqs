from dataclasses import dataclass


@dataclass
class SolverOption:
    pass


@dataclass
class FixedStep(SolverOption):
    dt: float


@dataclass
class AdaptiveStep(SolverOption):
    atol: float = 1e-8
    rtol: float = 1e-6
    max_steps: int = 100_000
    factor: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 5.0


@dataclass
class Euler(FixedStep):
    pass


@dataclass
class Dopri45(AdaptiveStep):
    pass
