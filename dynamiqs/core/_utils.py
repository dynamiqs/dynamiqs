from __future__ import annotations

from typing import get_args

from jax import numpy as jnp
from jaxtyping import ArrayLike

from ..solver import Solver
from ..time_array import TimeArray, _factory_constant
from .abstract_solver import AbstractSolver


def _astimearray(
    x: ArrayLike | TimeArray, dtype: jnp.complex64 | jnp.complex128
) -> TimeArray:
    if isinstance(x, get_args(ArrayLike)):
        return _factory_constant(x, dtype=dtype)
    else:
        return x


def get_solver_class(
    solvers: dict[Solver, AbstractSolver], solver: Solver
) -> AbstractSolver:
    if not isinstance(solver, tuple(solvers.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in solvers.keys())
        raise ValueError(
            f'Solver of type `{type(solver).__name__}` is not supported (supported'
            f' solver types: {supported_str}).'
        )
    return solvers[type(solver)]
