from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor


def smesolve(
    H: Tensor,
    Ls: List[Tensor],
    Ms: List[Tensor],
    etas: List[float],
    rho0: Tensor,
    tlist: np.ndarray,
    ntrajs: int,
    solver: str,
    options: Optional[Dict[str, Any]] = None,
) -> Tensor:
    raise NotImplementedError(
        'the Stochastic Master Equation solvers are not implemented yet, come '
        'back soon!'
    )
