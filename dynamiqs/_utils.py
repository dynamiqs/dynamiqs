from __future__ import annotations

from collections import namedtuple
from typing import Any

import diffrax as dx
from jax import numpy as jnp
from jaxtyping import Array

from .gradient import Autograd, Adjoint
from .utils import dag, isket


def type_str(type: Any) -> str:
    if type.__module__ in ('builtins', '__main__'):
        return f'`{type.__name__}`'
    else:
        return f'`{type.__module__}.{type.__name__}`'


def obj_type_str(x: Any) -> str:
    return type_str(type(x))


def split_complex(x: Array) -> Array:
    return jnp.stack((x.real, x.imag), axis=-1)


def merge_complex(x: Array) -> Array:
    return x[..., 0] + 1j * x[..., 1]


def check_time_array(x: Array, arg_name: str, allow_empty: bool = False):
    # check that a time array is valid (it must be a 1D array sorted in strictly
    # ascending order and containing only positive values)
    if x.ndim != 1:
        raise ValueError(
            f'Argument `{arg_name}` must be a 1D array, but is a {x.ndim}D array.'
        )
    if not allow_empty and len(x) == 0:
        raise ValueError(f'Argument `{arg_name}` must contain at least one element.')
    if not jnp.all(jnp.diff(x) > 0):
        raise ValueError(
            f'Argument `{arg_name}` must be sorted in strictly ascending order.'
        )
    if not jnp.all(x >= 0):
        raise ValueError(f'Argument `{arg_name}` must contain positive values only.')


def bexpect(O: Array, x: Array) -> Array:
    # batched over O
    if isket(x):
        return jnp.einsum('ij,...jk,kl->...', dag(x), O, x)  # <x|O|x>
    return jnp.einsum('bij,ji->b', O, x)  # tr(Ox)


SolverArgs = namedtuple('SolverArgs', ['save_states', 'exp_ops'])


def save_fn(_t, y, args: SolverArgs):
    res = {}
    if args.save_states:
        res['states'] = y
    if args.exp_ops is not None and len(args.exp_ops) > 0:
        y = merge_complex(y)
        exp = bexpect(args.exp_ops, y)
        res['expects'] = split_complex(exp)
    return res


def _get_solver_class(solver, solvers):
    if not isinstance(solver, tuple(solvers.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in solvers.keys())
        raise ValueError(
            f'Solver of type `{type(solver).__name__}` is not supported (supported'
            f' solver types: {supported_str}).'
        )
    solver_class = solvers[type(solver)]
    return solver_class


def _get_adjoint_class(gradient, solver):
    if gradient is None:
        return dx.RecursiveCheckpointAdjoint

    adjoints = {
        Autograd: dx.RecursiveCheckpointAdjoint,
        Adjoint: dx.BacksolveAdjoint,
    }
    if not isinstance(gradient, tuple(adjoints.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in adjoints.keys())
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

    gradient_class = adjoints[type(gradient)]
    return gradient_class
