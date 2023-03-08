from dataclasses import dataclass


@dataclass
class SolverOption:
    pass


@dataclass
class FixedStep(SolverOption):
    step: float


@dataclass
class VariableStep(SolverOption):
    atol: float = 1e-6
    rtol: float = 1e-6
    max_steps: int = 100_000


@dataclass
class Rouchon(FixedStep):
    pass


@dataclass
class RK4(FixedStep):
    pass


@dataclass
class DOPRI6(VariableStep):
    pass
