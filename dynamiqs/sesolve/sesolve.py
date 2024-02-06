from __future__ import annotations

import warnings
from functools import partial
from typing import Any, get_args

import diffrax as dx
import jax
from jax import numpy as jnp
from jaxtyping import ArrayLike

from .._utils import SolverArgs, _get_adjoint_class, _get_solver_class, save_fn
from ..gradient import Autograd, Gradient
from ..options import Options
from ..result import Result
from ..solver import Dopri5, Euler, Solver, _ODEAdaptiveStep, _stepsize_controller
from ..time_array import TimeArray, _factory_constant
from .schrodinger_term import SchrodingerTerm


@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def sesolve(
    H: ArrayLike | TimeArray,
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: ArrayLike | None = None,
    solver: Solver = Dopri5(),
    gradient: Gradient = Autograd(),
    options: dict[str, Any] | None = None,
) -> Result:
    # === convert arguments
    options = {} if options is None else options
    options = Options(**options)

    if isinstance(H, get_args(ArrayLike)):
        H = _factory_constant(H, dtype=options.cdtype)
    y0 = jnp.asarray(psi0, dtype=options.cdtype)
    tsave = jnp.asarray(tsave, dtype=options.rdtype)
    E = jnp.asarray(exp_ops, dtype=options.cdtype) if exp_ops is not None else None

    # === solver class
    solvers = {Dopri5: dx.Dopri5, Euler: dx.Euler}
    solver_class = _get_solver_class(solver, solvers)

    # === adjoint class
    adjoint_class = _get_adjoint_class(gradient, solver)

    # === stepsize controller
    stepsize_controller, dt = _stepsize_controller(solver)

    # === solve differential equation with diffrax
    term = SchrodingerTerm(H=H)
    max_steps = options.max_steps if isinstance(options, _ODEAdaptiveStep) else None

    # todo: remove once complex support is stabilized in diffrax
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)

        solution = dx.diffeqsolve(
            term,
            solver_class(),
            t0=tsave[0],
            t1=tsave[-1],
            dt0=dt,
            y0=y0,
            args=SolverArgs(save_states=options.save_states, E=E),
            saveat=dx.SaveAt(ts=tsave, fn=save_fn),
            stepsize_controller=stepsize_controller,
            adjoint=adjoint_class(),
            max_steps=max_steps,
        )

    # === get results
    ysave = solution.ys.get('ysave', None)
    Esave = solution.ys.get('Esave', None)
    if Esave is not None:
        Esave = Esave.swapaxes(-1, -2)

    return Result(tsave, solver, gradient, options, ysave, Esave)
