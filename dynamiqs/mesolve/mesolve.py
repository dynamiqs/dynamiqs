from __future__ import annotations

from typing import Any

from jaxtyping import ArrayLike

from dynamiqs.gradient import Gradient
from dynamiqs.solvers import Solver


def mesolve(
    H: ArrayLike,
    jump_ops: list[ArrayLike],
    rho0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver | None = None,
    gradient: Gradient | None = None,
    options: dict[str, Any] | None = None,
):
    pass
