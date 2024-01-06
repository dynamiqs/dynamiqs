from __future__ import annotations

from typing import Any

import diffrax as dx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from .lindblad_term import LindbladTerm
from .rouchon import Rouchon1Solver
from .._utils import save_fn, _get_solver_class, _get_adjoint_class
from ..gradient import Autograd, Gradient
from ..options import Options
from ..result import Result
from ..solver import Dopri5, Rouchon1, Solver, _stepsize_controller
from ..time_array import totime
from ..utils.utils import todm


def mesolve(
    H: ArrayLike,
    jump_ops: ArrayLike,
    rho0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: ArrayLike | None = None,
    solver: Solver = Dopri5(),
    gradient: Gradient = Autograd(),
    options: dict[str, Any] | None = None,
):
    # === options
    options = Options(solver=solver, gradient=gradient, options=options)

    # === solver class
    solvers = {Dopri5: dx.Dopri5, Rouchon1: Rouchon1Solver}
    solver_class = _get_solver_class(solver, solvers)

    # === adjoint class
    adjoint_class = _get_adjoint_class(gradient, solver)

    # === stepsize controller
    stepsize_controller, dt = _stepsize_controller(solver)

    # === solve differential equation with diffrax
    H = totime(H)
    Ls = totime(jump_ops)
    term = LindbladTerm(H=H, Ls=Ls)

    solution = dx.diffeqsolve(
        term,
        solver_class(),
        t0=tsave[0],
        t1=tsave[-1],
        dt0=dt,
        y0=todm(rho0),
        args=(options, exp_ops),
        saveat=dx.SaveAt(ts=tsave, fn=save_fn),
        stepsize_controller=stepsize_controller,
        adjoint=adjoint_class(),
    )

    ysave = None
    if options.save_states:
        ysave = solution.ys['states']

    Esave = None
    if "expects" in solution.ys:
        Esave = solution.ys['expects']
        Esave = jnp.stack(Esave, axis=0)

    return Result(
        options,
        ysave=ysave,
        Esave=Esave,
        tsave=solution.ts,
    )
