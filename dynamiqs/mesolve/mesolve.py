from __future__ import annotations

from typing import Any

import numpy as np
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
from ..utils.utils import todm


from ..time_array import TimeArray, totime


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
    print("BIP")
    Ls = format_L(jump_ops)
    term = LindbladTerm(H=H, Ls=Ls)

    print(H)
    print(Ls)
    print("AAAA")

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


def format_L(L: list[ArrayLike | TimeArray]) -> list[TimeArray]:
    # Format a list of tensors of individual shape (n, n) or (?, n, n) into a single
    # batched tensor of shape (nL, bL, n, n). An error is raised if all batched
    # dimensions `?` are not the same.

    L = [totime(x) for x in L]
    n = L[0](0.0).shape[-1]
    L = [x.reshape(-1, n, n) for x in L]  # [(?, n, n)] with ? = 1 if not batched

    bL = common_batch_size([x.shape[0] for x in L])
    if bL is None:
        L_shapes = [tuple(x.shape) for x in L]
        raise ValueError(
            'Argument `jump_ops` should be a list of 2D arrays or 3D arrays with the'
            ' same batch size, but got a list of arrays with incompatible shapes'
            f' {L_shapes}.'
        )

    res = []
    for x in L:
        if x.ndim == 3:
            L.append(x)
        elif x.ndim == 2:
            L.append(x.repeat(bL, 0))
        else:
            raise Exception(f'Unexpected dimension {x.ndim}.')

    return res


def common_batch_size(dims: list[int]) -> int | None:
    # If `dims` is a list with two values 1 and x, returns x. If it contains only
    # 1, returns 1. Otherwise, returns `None`.
    bs = np.unique(dims)
    if (1 not in bs and len(bs) > 1) or len(bs) > 2:
        return None
    return bs.max().item()
