from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..core._utils import _astimearray, compute_vmap, get_solver_class
from ..gradient import Gradient
from ..options import Options
from ..result import Result
from ..solver import Dopri5, Euler, Solver
from ..time_array import TimeArray
from ..utils.array_types import cdtype
from ..utils.utils import todm
from .mediffrax import MEDopri5, MEEuler


@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def mesolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Dopri5(),
    gradient: Gradient | None = None,
    options: Options = Options(),
):
    # === vectorize function
    # we vectorize over H, jump_ops and psi0, all other arguments are not vectorized
    jump_ops_ndim = _astimearray(jump_ops[0]).ndim + 1
    is_batched = (
        H.ndim > 2,
        jump_ops_ndim > 3,  # todo: this is a temporary fix
        psi0.ndim > 2,
        False,
        False,
        False,
        False,
        False,
    )
    # the result is vectorized over ysave and Esave
    out_axes = Result(None, None, None, None, 0, 0)

    f = compute_vmap(_mesolve, options.cartesian_batching, is_batched, out_axes)

    # === apply vectorized function
    return f(H, jump_ops, psi0, tsave, exp_ops, solver, gradient, options)


def _mesolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Dopri5(),
    gradient: Gradient | None = None,
    options: Options = Options(),
) -> Result:
    # === convert arguments
    H = _astimearray(H)
    Ls = [_astimearray(L) for L in jump_ops]
    y0 = jnp.asarray(psi0, dtype=cdtype())
    y0 = todm(y0)
    ts = jnp.asarray(tsave)
    Es = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None

    # === select solver class
    solvers = {Euler: MEEuler, Dopri5: MEDopri5}
    solver_class = get_solver_class(solvers, solver)

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === init solver
    solver = solver_class(ts, y0, H, Es, solver, gradient, options, Ls)

    # === run solver
    result = solver.run()

    # === return result
    return result
