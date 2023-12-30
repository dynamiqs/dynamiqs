from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import PyTree

from ..solver import Euler, Propagator, Solver
from ..solvers.options import Options
from ..time_array import ConstantTimeArray, TimeArray
from .euler import SEEuler
from .propagator import SEPropagator


def sesolve(
    H: ArrayLike | TimeArray,
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver | None = None,
    # gradient: None = None,
    options: dict[str, Any] | None = None,
) -> PyTree:
    # === default solver
    if solver is None:
        pass

    # === options
    options = Options(solver=solver, options=options)

    # === solver class
    solvers = {
        Propagator: SEPropagator,
        Euler: SEEuler,
        # BackwardEuler: SEBackwardEuler,
        # Dopri5: SEDormandPrince5,
    }
    if not isinstance(solver, tuple(solvers.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in solvers.keys())
        raise ValueError(
            f'Solver of type `{type(solver).__name__}` is not supported (supported'
            f' solver types: {supported_str}).'
        )
    SOLVER_CLASS = solvers[type(solver)]

    # === convert H
    if not isinstance(H, TimeArray):
        H = jnp.asarray(H, dtype=options.cdtype)  # (n, n)
        H = ConstantTimeArray(H)

    # === convert y0
    y0 = jnp.asarray(psi0, dtype=options.cdtype)  # (n, 1)

    # === convert tsave
    ts = jnp.asarray(tsave, dtype=options.rdtype)

    # === convert E
    if exp_ops is None:
        exp_ops = []
    E = jnp.asarray(exp_ops, dtype=options.cdtype)  # (nE, n, n)

    # === define the solver
    solver = SOLVER_CLASS(ts, y0, H, E, options)

    # === compute the result
    result = solver.run()

    return result
