from __future__ import annotations

import equinox as eqx


class Options(eqx.Module):
    save_states: bool
    verbose: bool
    cartesian_batching: bool

    def __init__(
        self,
        *,
        save_states: bool = True,
        verbose: bool = True,
        cartesian_batching: bool = True,
    ):
        # save_states (bool, optional): If `True`, the state is saved at every
        #     time value. If `False`, only the final state is stored and returned.
        #     Defaults to `True`.
        self.save_states = save_states
        self.verbose = verbose
        self.cartesian_batching = cartesian_batching
