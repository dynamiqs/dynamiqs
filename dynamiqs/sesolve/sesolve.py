from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..core._utils import _astimearray, compute_vmap, get_solver_class
from ..options import Options
from ..result import Result
from ..solver import Dopri5, Dopri8, Euler, Propagator, Solver, Tsit5
from ..time_array import TimeArray
from ..utils.array_types import cdtype
from .sediffrax import SEDopri5, SEDopri8, SEEuler, SETsit5
from .sepropagator import SEPropagator


@partial(jax.jit, static_argnames=('solver', 'options'))
def sesolve(
    H: ArrayLike | TimeArray,
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Tsit5(),
    options: Options = Options(),
):
    # === vectorize function
    # we vectorize over H and psi0, all other arguments are not vectorized
    is_batched = (H.ndim > 2, psi0.ndim > 2, False, False, False, False)
    # the result is vectorized over ysave and Esave
    out_axes = Result(None, None, None, 0, 0)
    f = compute_vmap(_sesolve, options.cartesian_batching, is_batched, out_axes)

    # === apply vectorized function
    return f(H, psi0, tsave, exp_ops, solver, options)


def _sesolve(
    H: ArrayLike | TimeArray,
    psi0: ArrayLike,
    tsave: ArrayLike,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Tsit5(),
    options: Options = Options(),
) -> Result:
    # === convert arguments
    H = _astimearray(H)
    y0 = jnp.asarray(psi0, dtype=cdtype())
    ts = jnp.asarray(tsave)
    Es = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None

    # === select solver class
    solvers = {
        Euler: SEEuler,
        Dopri5: SEDopri5,
        Dopri8: SEDopri8,
        Tsit5: SETsit5,
        Propagator: SEPropagator,
    }
    solver_class = get_solver_class(solvers, solver)

    # === init solver
    solver = solver_class(ts, y0, H, Es, solver, options)

    # === run solver
    result = solver.run()

    # === return result
    return result
