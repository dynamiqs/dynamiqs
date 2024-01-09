from __future__ import annotations
from typing import Callable

import diffrax as dx
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Scalar

from .._utils import merge_complex, split_complex
from ..time_array import TimeArray
from ..utils.utils import dag


class LindbladTerm(dx.ODETerm):
    H: TimeArray  # (n, n)
    Ls: list[TimeArray]  # (nL, n, n)
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree]

    def __init__(self, H: TimeArray, Ls: TimeArray):
        self.H = H
        self.Ls = Ls

    def vector_field(self, t: Scalar, rho: PyTree, _args: PyTree):
        rho = merge_complex(rho)
        H_t = self.H(t)
        Ls_t = jnp.stack([L(t) for L in self.Ls])
        Hnh_t = Hnh(H_t, Ls_t)
        out = -1j * Hnh_t @ rho + 0.5 * jnp.sum(Ls_t @ rho @ dag(Ls_t), axis=0)
        return split_complex(out + dag(out))

    @property
    def Id(self) -> Array:
        return jnp.eye(self.H.shape[-1], dtype=self.H.dtype)


def Hnh(H_t: Array, Ls_t: list[Array]) -> Array:
    return H_t - 0.5j * jnp.sum(dag(Ls_t) @ Ls_t, axis=0)
