from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor


def ssolve(
    H: Tensor,
    rho0: Tensor,
    tlist: np.ndarray,
    solver: str,
    options: Optional[Dict[str, Any]] = None,
) -> Tensor:
    raise NotImplementedError(
        'the Schrödinger Master Equation solvers are not implemented yet, come '
        'back soon!'
    )
