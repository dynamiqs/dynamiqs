from __future__ import annotations

import warnings

from jaxtyping import ArrayLike, PRNGKeyArray

from ...gradient import Gradient
from ...method import LowRank, Method, Tsit5
from ...options import Options
from ...qarrays.qarray import QArrayLike
from ...result import MESolveResult
from ...time_qarray import TimeQArray
from .mesolve import mesolve


def mesolve_lr(
    H: QArrayLike | TimeQArray,
    jump_ops: list[QArrayLike | TimeQArray],
    rho0: QArrayLike,
    tsave: ArrayLike,
    *,
    M: int,
    exp_ops: list[QArrayLike] | None = None,
    method: Method = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
    normalize_each_eval: bool = True,
    linear_solver: str = 'lineax',
    save_factors_only: bool = False,
    eps_init: float | None = None,
    key: PRNGKeyArray | None = None,
) -> MESolveResult:
    """Deprecated wrapper for low-rank Lindblad master equation integration.

    Use `dq.mesolve(..., method=dq.method.LowRank(...))` instead.
    """
    warnings.warn(
        '`dq.mesolve_lr()` is deprecated. Use `dq.mesolve()` with '
        '`method=dq.method.LowRank(...)` instead.',
        DeprecationWarning,
        stacklevel=2,
    )
    lr_method = LowRank(
        M=M,
        ode_method=method,
        normalize_each_eval=normalize_each_eval,
        linear_solver=linear_solver,
        save_factors_only=save_factors_only,
        eps_init=eps_init,
        key=key,
    )
    return mesolve(
        H,
        jump_ops,
        rho0,
        tsave,
        exp_ops=exp_ops,
        method=lr_method,
        gradient=gradient,
        options=options,
    )
