from __future__ import annotations

from typing import Callable

import equinox as eqx
from jax import Array
from jaxtyping import PyTree, Scalar


class Options(eqx.Module):
    save_states: bool = True
    verbose: bool = True
    cartesian_batching: bool = True
    t0: Scalar | None = None  # defaults to tsave[0]
    save_fn: Callable[[Array], PyTree] | None = None
