from __future__ import annotations

import warnings
from typing import Any

import diffrax as dx
from jax import numpy as jnp
from jaxtyping import ArrayLike

from .._utils import SolverArgs, _get_adjoint_class, _get_solver_class, save_fn
from ..gradient import Autograd, Gradient
from ..options import Options
from ..result import Result
from ..solver import Dopri5, Euler, Solver, _ODEAdaptiveStep, _stepsize_controller
from ..time_array import totime
from .schrodinger_term import SchrodingerTerm


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
    solvers = {Dopri5: dx.Dopri5, Euler: dx.Euler}
    solver_class = _get_solver_class(solver, solvers)

    # === adjoint class
    adjoint_class = _get_adjoint_class(gradient, solver)

    # === stepsize controller
    stepsize_controller, dt = _stepsize_controller(solver)

    # === solve differential equation with diffrax
    H = totime(H)
    term = SchrodingerTerm(H=H)
    if exp_ops is not None:
        exp_ops = jnp.asarray(exp_ops)
    else:
        exp_ops = jnp.empty(0)

    # todo: remove once complex support is stabilized in diffrax
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)

        solution = dx.diffeqsolve(
            term,
            solver_class(),
            t0=tsave[0],
            t1=tsave[-1],
            dt0=dt,
            y0=psi0,
            args=SolverArgs(save_states=options.save_states, exp_ops=exp_ops),
            saveat=dx.SaveAt(ts=tsave, fn=save_fn),
            stepsize_controller=stepsize_controller,
            adjoint=adjoint_class(),
            max_steps=(
                options.max_steps if isinstance(options, _ODEAdaptiveStep) else None
            ),
            progress_meter=options.progress_bar,
        )

    # === get results
    ysave = None
    if options.save_states:
        ysave = solution.ys['states']

    Esave = None
    if 'expects' in solution.ys:
        Esave = jnp.stack(solution.ys['expects'].T, axis=0)

    return Result(
        options,
        ysave=ysave,
        Esave=Esave,
        tsave=solution.ts,
    )
