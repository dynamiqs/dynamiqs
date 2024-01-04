from __future__ import annotations

from typing import Any

import diffrax
from diffrax import diffeqsolve
from jaxtyping import ArrayLike
from jax import numpy as jnp
from .._utils import split_complex, merge_complex, bexpect
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
    def f(_t, psi, _args):
        psi = merge_complex(psi)
        res = -1j * H @ psi
        res = split_complex(res)
        return res

    def save(_t, psi, _args):
        res = {}
        if options.save_states:
            res["states"] = psi

        psi = merge_complex(psi)
        res["expects"] = tuple([split_complex(bexpect(op, psi)) for op in exp_ops])
        return res

    solution = diffeqsolve(
        diffrax.ODETerm(f),
        solver_class(),
        t0=tsave[0],
        t1=tsave[-1],
        dt0=tsave[1],
        y0=split_complex(psi0),
        saveat=diffrax.SaveAt(ts=tsave, fn=save),
    )

    ysave = None
    if options.save_states:
        ysave = solution.ys["states"]
        ysave = merge_complex(ysave)

    Esave = None
    if len(exp_ops) > 0:
        Esave = solution.ys["expects"]
        Esave = jnp.stack(Esave, axis=0)
        Esave = merge_complex(Esave)

    return Result(
        options,
        ysave=ysave,
        Esave=Esave,
        tsave=solution.ts,
    )
