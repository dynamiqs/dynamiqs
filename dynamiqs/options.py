from __future__ import annotations

from typing import Callable

import equinox as eqx
from jax import Array
from jaxtyping import PyTree, Scalar


class Options(eqx.Module):
    """
    Options for the solver.

    Parameters:
        save_states (bool): Whether to save all the states. If `False` only the last
            state is saved.
        verbose (bool): Whether to print progress information.
        cartesian_batching (bool): Whether to perform cartesian product on batched
            arguments. If `True` the solver will be called with every combination
            of batched arguments. If `False` the solver will be called with zipped
            batched arguments.
        t0 (Scalar, optional): Initial simulation time. Defaults to `tsave[0]`.
        save_fn (Callable[[Array], PyTree], optional): A function to save
            additional arbitrary data. The function should take the state
            as input and return a pytree of arrays. Defaults to `None`.
    """

    save_states: bool = True
    verbose: bool = True
    cartesian_batching: bool = True
    t0: Scalar | None = None  # defaults to tsave[0]
    save_fn: Callable[[Array], PyTree] | None = None
