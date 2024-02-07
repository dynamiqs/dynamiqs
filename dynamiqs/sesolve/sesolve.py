from __future__ import annotations

from functools import partial

import jax
from jax import numpy as jnp
from jaxtyping import ArrayLike

from ..core._utils import _astimearray, get_solver_class
from ..gradient import Gradient
from ..options import Options
from ..result import Result
from ..solver import Dopri5, Euler, Solver
from ..time_array import TimeArray
from .sediffrax import SEDopri5, SEEuler


@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def sesolve(
    H: ArrayLike | TimeArray,
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: ArrayLike | None = None,
    solver: Solver = Dopri5(),
    gradient: Gradient | None = None,
    options: Options = Options(),
) -> Result:
    # === convert arguments
    H = _astimearray(H, dtype=options.cdtype)
    y0 = jnp.asarray(psi0, dtype=options.cdtype)
    ts = jnp.asarray(tsave, dtype=options.rdtype)
    Es = jnp.asarray(exp_ops, dtype=options.cdtype) if exp_ops is not None else None

    # === select solver class
    solvers = {Euler: SEEuler, Dopri5: SEDopri5}
    solver_class = get_solver_class(solvers, solver)

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === init solver
    solver = solver_class(ts, y0, H, Es, solver, gradient, options)

    # === run solver
    result = solver.run()

    # === return result
    return result
