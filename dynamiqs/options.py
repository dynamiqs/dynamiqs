from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
from jax import Array
from jaxtyping import PyTree, Scalar


class Options(eqx.Module):
    save_states: bool = True
    verbose: bool = True
    cartesian_batching: bool = True
    t0: Scalar | None = None  # defaults to tsave[0]
    save_fun: Callable[[Array], PyTree] | None = None

    def __post_init__(self):
        if self.save_fun is not None:
            # use `jax.tree_util.Partial` to make `save_fun` a valid Pytree
            self.save_fun = jax.tree_util.Partial(self.save_fun)
