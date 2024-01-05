from __future__ import annotations

from typing import Any

import diffrax as dx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from .._utils import save_fn
from ..gradient import Adjoint, Autograd, Gradient
from ..options import Options
from ..result import Result
from ..solver import Dopri5, Rouchon1, Solver, _stepsize_controller
from ..time_array import totime
from ..utils.utils import todm
from .lindblad_term import LindbladTerm
from .rouchon import Rouchon1Solver


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
    save_expects = exp_ops is not None and len(exp_ops) > 0
    options = Options(
        solver=solver, gradient=gradient, options=options, save_expects=save_expects
    )

    # === solver class
    solvers = {Dopri5: dx.Dopri5, Rouchon1: Rouchon1Solver}
    if not isinstance(solver, tuple(solvers.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in solvers.keys())
        raise ValueError(
            f'Solver of type `{type(solver).__name__}` is not supported (supported'
            f' solver types: {supported_str}).'
        )
    solver_class = solvers[type(solver)]

    # === adjoint class
    gradients = {
        Autograd: dx.RecursiveCheckpointAdjoint,
        Adjoint: dx.BacksolveAdjoint,
    }
    if not isinstance(gradient, tuple(gradients.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in gradients.keys())
        raise ValueError(
            f'Gradient of type `{type(gradient).__name__}` is not supported'
            f' (supported gradient types: {supported_str}).'
        )
    elif not solver.supports_gradient(gradient):
        support_str = ', '.join(f'`{x.__name__}`' for x in solver.SUPPORTED_GRADIENT)
        raise ValueError(
            f'Solver `{type(solver).__name__}` does not support gradient'
            f' `{type(gradient).__name__}` (supported gradient types: {support_str}).'
        )
    adjoint_class = gradients[type(gradient)]

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
    if options.save_expects:
        Esave = solution.ys['expects']
        Esave = jnp.stack(Esave, axis=0)

    return Result(
        options,
        ysave=ysave,
        Esave=Esave,
        tsave=solution.ts,
    )
