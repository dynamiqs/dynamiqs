from dataclasses import dataclass


@dataclass
class SolverOption:
    pass


@dataclass
class FixedStep(SolverOption):
    dt: float


@dataclass
class AdaptativeStep(SolverOption):
    atol: float = 1e-6
    rtol: float = 1e-6
    max_steps: int = 100_000


@dataclass
class Rouchon(FixedStep):
    order: int = 2


@dataclass
class RK4(FixedStep):
    pass


@dataclass
class DOPRI6(AdaptativeStep):
    pass
