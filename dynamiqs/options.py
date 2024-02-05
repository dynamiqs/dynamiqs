from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .utils.array_types import get_cdtype


class Options(eqx.Module):
    save_states: bool
    verbose: bool
    _double_precision: bool
    cartesian_batching: bool

    def __init__(
        self,
        *,
        save_states: bool = True,
        verbose: bool = True,
        dtype: jnp.complex64 | jnp.complex128 | None = None,
        cartesian_batching: bool = True,
    ):
        # save_states (bool, optional): If `True`, the state is saved at every
        #     time value. If `False`, only the final state is stored and returned.
        #     Defaults to `True`.
        # dtype (jax.numpy.dtype, optional): Complex data type to which all
        #     complex-valued arrays are converted. `tsave` is also converted to a real
        #     data type of the corresponding precision.
        self.save_states = save_states
        self.verbose = verbose
        cdtype = get_cdtype(dtype)
        self._double_precision = cdtype == jnp.complex128
        self.cartesian_batching = cartesian_batching

    @property
    def rdtype(self) -> jnp.float32 | jnp.float64:
        return jnp.float64 if self._double_precision else jnp.float32

    @property
    def cdtype(self) -> jnp.complex64 | jnp.complex128:
        return jnp.complex128 if self._double_precision else jnp.complex64
