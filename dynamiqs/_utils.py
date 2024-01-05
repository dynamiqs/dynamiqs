from __future__ import annotations

from typing import Any

from jax import numpy as jnp
from jaxtyping import Array
import diffrax as dx

from gradient import Autograd, Adjoint
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


def check_time_array(x: Array, arg_name: str, allow_empty=False):
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
    if isket(x):
        return jnp.einsum('...ij,jk,...kl->...', dag(x), O, x)  # <x|O|x>
    return jnp.einsum('ij,bji->b', O, x)  # tr(Ox)


def _get_gradient_class(gradient, solver):
    gradients = {
        Autograd: dx.RecursiveCheckpointAdjoint,
        Adjoint: dx.BacksolveAdjoint,
    }
    if gradient is None:
        pass
    elif not isinstance(gradient, tuple(gradients.keys())):
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
    if gradient is not None:
        gradient_class = gradients[type(gradient)]
    else:
        gradient_class = None
    return gradient_class


def _get_solver_class(solver, solvers):
    if not isinstance(solver, tuple(solvers.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in solvers.keys())
        raise ValueError(
            f'Solver of type `{type(solver).__name__}` is not supported (supported'
            f' solver types: {supported_str}).'
        )
    solver_class = solvers[type(solver)]
    return solver_class
