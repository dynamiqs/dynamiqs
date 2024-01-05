from typing import Callable

import diffrax as dx
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Scalar

from ..time_array import TimeArray
from ..utils.utils import dag


class LindbladTerm(dx.ODETerm):
    H: TimeArray  # (n, n)
    Ls: TimeArray  # (nL, n, n)
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree]

    def __init__(self, H: TimeArray, Ls: TimeArray):
        self.H = H
        self.Ls = Ls

    def vector_field(self, t: Scalar, rho: PyTree, _args: PyTree):
        Ls_t = self.Ls(t)
        out = -1j * self.Hnh(t) @ rho + 0.5 * jnp.sum(Ls_t @ rho @ dag(Ls_t), axis=0)
        return out + dag(out)

    def Hnh(self, t: Scalar) -> Array:
        Ls_t = self.Ls(t)
        return self.H(t) - 0.5j * jnp.sum(dag(Ls_t) @ Ls_t, axis=0)

    @property
    def Id(self) -> Array:
        return jnp.eye(self.H.shape[-1], dtype=self.H.dtype)
