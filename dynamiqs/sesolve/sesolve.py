from __future__ import annotations

from typing import Any

import diffrax as dx
from jax import numpy as jnp
from jaxtyping import ArrayLike

from .schrodinger_term import SchrodingerTerm
from .._utils import (
    save_fn,
    _get_adjoint_class,
    _get_solver_class,
    SolverArgs,
    merge_complex,
    split_complex,
)
from ..gradient import Autograd, Gradient
from ..options import Options
from ..result import Result
from ..solver import Dopri5, Solver, _stepsize_controller
from ..time_array import totime


def sesolve(
    H: ArrayLike,
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: ArrayLike | None = None,
    solver: Solver = Dopri5(),
    gradient: Gradient = Autograd(),
    options: dict[str, Any] | None = None,
) -> Result:
    # === options
    options = Options(solver=solver, gradient=gradient, options=options)

    # === solver class
    solvers = {Dopri5: dx.Dopri5}
    solver_class = _get_solver_class(solver, solvers)

    # === adjoint class
    adjoint_class = _get_adjoint_class(gradient, solver)

    # === stepsize controller
    stepsize_controller, dt = _stepsize_controller(solver)

    # === solve differential equation with diffrax
    exp_ops = jnp.array(exp_ops)
    H = totime(H)

    term = SchrodingerTerm(H=H)

    solution = dx.diffeqsolve(
        term,
        solver_class(),
        t0=tsave[0],
        t1=tsave[-1],
        dt0=dt,
        y0=split_complex(psi0),
        args=SolverArgs(save_states=options.save_states, exp_ops=exp_ops),
        saveat=dx.SaveAt(ts=tsave, fn=save_fn),
        stepsize_controller=stepsize_controller,
        adjoint=adjoint_class(),
    )

    # === get results
    ysave = None
    if options.save_states:
        ysave = merge_complex(solution.ys['states'])

    Esave = None
    if "expects" in solution.ys:
        Esave = merge_complex(solution.ys['expects']).T
        Esave = jnp.stack(Esave, axis=0)

    return Result(
        options,
        ysave=ysave,
        Esave=Esave,
        tsave=solution.ts,
    )
