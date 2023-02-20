from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class BaseSolverOption:
    H: torch.Tensor
    jump_ops: List[torch.Tensor]
    expect_ops: List[torch.Tensor]
    t_save: np.array


@dataclass
class FixedStepOption(BaseSolverOption):
    pass


@dataclass
class VariableStepOption(BaseSolverOption):
    atol: float = 1e-6
    rtol: float = 1e-6
    max_steps: int = 100_000
