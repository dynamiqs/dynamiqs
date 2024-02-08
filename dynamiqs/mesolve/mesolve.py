from __future__ import annotations

import warnings
from typing import Any

import diffrax as dx
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from .._utils import SolverArgs, _get_adjoint_class, _get_solver_class, save_fn
from ..gradient import Autograd, Gradient
from ..options import Options
from ..result import Result
from ..solver import Dopri5, Rouchon1, Solver, _stepsize_controller
from ..time_array import TimeArray, totime
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
    Ls = [totime(L) for L in jump_ops]
    Ls = format_L(Ls)
    term = LindbladTerm(H=H, Ls=Ls)
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
            y0=todm(rho0),
            args=SolverArgs(save_states=options.save_states, exp_ops=exp_ops),
            saveat=dx.SaveAt(ts=tsave, fn=save_fn),
            stepsize_controller=stepsize_controller,
            adjoint=adjoint_class(),
            progress_meter=options.progress_bar,
        )

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


def format_L(Ls: list[TimeArray]) -> list[TimeArray]:
    # Format a list of TimeArrays of individual shape (n, n) or (?, n, n) into a list of
    # TimeArrays of shape (bL, n, n) where bL is a common batch size. An error is
    # raised if all batched dimensions `?` do not match.
    n = Ls[0].shape[-1]
    Ls = [L.reshape(-1, n, n) for L in Ls]  # [(?, n, n)] with ? = 1 if not batched

    bL = common_batch_size([L.shape[0] for L in Ls])
    if bL is None:
        L_shapes = [tuple(L.shape) for L in Ls]
        raise ValueError(
            'Argument `jump_ops` should be a list of 2D arrays or 3D arrays with the'
            ' same batch size, but got a list of arrays with incompatible shapes'
            f' {L_shapes}.'
        )

    Ls_formatted = []
    for L in Ls:
        if L.ndim == 3:
            Ls_formatted.append(L if bL > 1 else L.reshape(n, n))
        elif L.ndim == 2:
            Ls_formatted.append(L.repeat(bL, 0) if bL > 1 else L)
        else:
            raise Exception(f'Unexpected dimension {L.ndim}.')

    return Ls_formatted


def common_batch_size(dims: list[int]) -> int | None:
    # If `dims` is a list with two values 1 and x, returns x. If it contains only
    # 1, returns 1. Otherwise, returns `None`.
    bs = np.unique(dims)
    if (1 not in bs and len(bs) > 1) or len(bs) > 2:
        return None
    return bs.max().item()
