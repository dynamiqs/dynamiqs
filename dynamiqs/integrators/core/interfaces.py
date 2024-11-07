import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import Scalar

from ...options import Options
from ...time_array import TimeArray


class OptionsInterface(eqx.Module):
    options: Options


class SEInterface(eqx.Module):
    """Interface for the Schrödinger equation."""

    H: TimeArray


class MEInterface(eqx.Module):
    """Interface for the Lindblad master equation."""

    H: TimeArray
    Ls: list[TimeArray]

    def L(self, t: Scalar) -> Array:
        return jnp.stack([L(t) for L in self.Ls])  # (nLs, n, n)


class SolveInterface(eqx.Module):
    Es: Array
