from typing import Any, Dict, Optional

import numpy as np
import torch


def ssolve(
    H: torch.Tensor,
    rho0: torch.Tensor,
    tlist: np.ndarray,
    solver: str,
    options: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    raise NotImplementedError(
        'the Schrödinger Master Equation solvers are not implemented yet, come '
        'back soon!'
    )
