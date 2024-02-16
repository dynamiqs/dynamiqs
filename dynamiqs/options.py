from __future__ import annotations

import equinox as eqx
from jaxtyping import Scalar

__all__ = ['Options']

class Options(eqx.Module):
    save_states: bool = True
    verbose: bool = True
    cartesian_batching: bool = True
    t0: Scalar | None = None  # defaults to tsave[0]

    def __init__(self, save_states: bool = True, verbose: bool = True, cartesian_batching: bool = True, t0: Scalar | None = None):
        """Generic options for solvers.

        Args:
            save_states: Whether to save the states at each time step. If `False`, only
                the final state is returned.
            verbose: Whether to print information about the solver integration.
            cartesian_batching: Whether to use the cartesian batching or not. Cartesian
                batching means batching only once over multiple batched arrays.
                Otherwise, each batched array is batched separately.
            t0: Initial time. If `None`, defaults to the first time in `tsave`.
        """
        self.save_states = save_states
        self.verbose = verbose
        self.cartesian_batching = cartesian_batching
        self.t0 = t0
