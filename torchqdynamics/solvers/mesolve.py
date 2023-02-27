from dataclasses import dataclass
from typing import List

import torch


@dataclass
class MESolverOption:
    jump_ops: List[torch.Tensor]


@dataclass
class FixedStepOption(MESolverOption):
    step: float


@dataclass
class VariableStepOption(MESolverOption):
    atol: float = 1e-6
    rtol: float = 1e-6
    max_steps: int = 100_000


@dataclass
class Rouchon(FixedStepOption):
    pass


@dataclass
class RK4(FixedStepOption):
    pass


@dataclass
class DOPRI6(VariableStepOption):
    pass
