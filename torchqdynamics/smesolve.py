from __future__ import annotations

from typing import Any

import numpy as np
from torch import Tensor


def smesolve(
    H: Tensor,
    Ls: list[Tensor],
    Ms: list[Tensor],
    etas: list[float],
    rho0: Tensor,
    tlist: np.ndarray,
    ntrajs: int,
    solver: str,
    options: dict[str, Any] | None = None,
) -> Tensor:
    raise NotImplementedError(
        'the Stochastic Master Equation solvers are not implemented yet, come '
        'back soon!'
    )
