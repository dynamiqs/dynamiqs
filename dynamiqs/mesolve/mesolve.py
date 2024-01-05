from __future__ import annotations

from typing import Any

import diffrax as dx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from .._utils import bexpect, _get_gradient_class, _get_solver_class
from ..gradient import Adjoint, Autograd, Gradient
from ..options import Options
from ..result import Result
from ..solver import Rouchon1, Solver, _stepsize_controller
from .lindblad_term import LindbladTerm
from .rouchon import Rouchon1Solver


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
    # === default solver
    if solver is None:
        solver = Rouchon1(dt=0.01)

    # === options
    options = Options(solver=solver, gradient=gradient, options=options)

    # === solver class
    solvers = {Rouchon1: Rouchon1Solver}
    solver_class = _get_solver_class(solver, solvers)

    # === gradient class
    gradient_class = _get_gradient_class(gradient, solver)

    # === stepsize controller
    stepsize_controller, dt = _stepsize_controller(solver)

    # === solve differential equation with diffrax
    term = LindbladTerm(H=H, Ls=jnp.stack(jump_ops))

    def save(_t, rho, _args):
        res = {}
        if options.save_states:
            res['states'] = rho

        # TODO : use vmap ?
        if exp_ops is not None:
            res['expects'] = tuple([bexpect(op, rho) for op in exp_ops])
        return res

    solution = dx.diffeqsolve(
        term,
        solver_class(),
        t0=tsave[0],
        t1=tsave[-1],
        dt0=dt,
        y0=rho0,
        saveat=dx.SaveAt(ts=tsave, fn=save),
        stepsize_controller=stepsize_controller,
        adjoint=(
            gradient_class()
            if gradient_class is not None
            else dx.RecursiveCheckpointAdjoint()
        ),
    )

    ysave = None
    if options.save_states:
        ysave = solution.ys['states']
        ysave = ysave

    Esave = None
    if exp_ops is not None and len(exp_ops) > 0:
        Esave = solution.ys['expects']
        Esave = jnp.stack(Esave, axis=0)
        Esave = Esave

    return Result(
        options,
        ysave=ysave,
        Esave=Esave,
        tsave=solution.ts,
    )
