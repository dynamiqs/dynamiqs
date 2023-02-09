from typing import Any, Dict, List, Optional

import numpy as np
import torch


def smesolve(
    H: torch.Tensor,
    Ls: List[torch.Tensor],
    Ms: List[torch.Tensor],
    etas: List[float],
    rho0: torch.Tensor,
    tlist: np.ndarray,
    ntrajs: int,
    solver: str,
    options: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    raise NotImplementedError(
        'the Stochastic Master Equation solvers are not implemented yet, come '
        'back soon!'
    )
