from __future__ import annotations

from typing import Any

import diffrax
from diffrax import diffeqsolve
from jaxtyping import ArrayLike

from .._utils import split_complex, merge_complex
from ..gradient import Gradient
from ..options import Options
from ..result import Result
from ..solvers import Dopri5, Solver


def sesolve(
    H: ArrayLike,
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver | None = None,
    gradient: Gradient | None = None,
    options: dict[str, Any] | None = None,
) -> Result:
    # === default solver
    solver = Dopri5()

    # === options
    options = Options(solver=solver, gradient=gradient, options=options)

    # === solver class
    solvers = {Dopri5: diffrax.Dopri5}
    if not isinstance(solver, tuple(solvers.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in solvers.keys())
        raise ValueError(
            f'Solver of type `{type(solver).__name__}` is not supported (supported'
            f' solver types: {supported_str}).'
        )
    solver_class = solvers[type(solver)]

    # === solve differential equation with diffrax
    def f(t, psi, args):
        psi = merge_complex(psi)
        res = -1j * H @ psi
        res = split_complex(res)
        return res

    def save(t, psi, args):
        res = tuple()
        psi = merge_complex(psi)
        if options.save_states:
            res += (psi,)
        for exp_op in exp_ops:
            res += bexpect(exp_op, psi)

    solution = diffeqsolve(
        diffrax.ODETerm(f),
        solver_class(),
        t0=tsave[0],
        t1=tsave[-1],
        dt0=tsave[1],
        y0=split_complex(psi0),
        saveat=diffrax.SaveAt(ts=tsave),
    )

    return Result(
        options,
        ysave=merge_complex(solution.ys),
        tsave=solution.ts,
        Esave=None,
    )
