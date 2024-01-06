import functools
from typing import Callable

import diffrax as dx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Scalar

from ..time_array import TimeArray, ConstantTimeArray
from ..utils.utils import dag


class LindbladTerm(dx.ODETerm):
    H: TimeArray  # (n, n)
    Ls: TimeArray  # (nL, n, n)
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree]

    def __init__(self, H: TimeArray, Ls: TimeArray):
        self.constant_H = isinstance(self.H, ConstantTimeArray)
        # rewrite but shorter
        self.H = H(0.0) if self.constant_H else H

        self.constant_Ls = isinstance(self.Ls, ConstantTimeArray)
        self.Ls = Ls(0.0) if self.constant_Ls else Ls

        self.constant_Hnh = self.constant_H and self.constant_Ls
        if self.constant_Hnh:
            self.Hnh = LindbladTerm.Hnh(self.H, self.Ls)
        elif self.constant_H:
            self.Hnh = lambda t: LindbladTerm.Hnh(self.H, self.Ls(t))
        elif self.constant_Ls:
            self.Hnh = lambda t: LindbladTerm.Hnh(self.H(t), self.Ls)
        else:
            self.Hnh = lambda t: LindbladTerm.Hnh(self.H(t), self.Ls(t))

    @functools.partial(jax.jit, static_argnums=0)
    def vector_field(self, t: Scalar, rho: PyTree, _args: PyTree):
        Ls_t = self.Ls if self.constant_Ls else self.Ls(t)
        Hnh = self.Hnh if self.constant_Hnh else self.Hnh(t)

        out = -1j * Hnh @ rho + 0.5 * jnp.sum(Ls_t @ rho @ dag(Ls_t), axis=0)
        return out + dag(out)

    @staticmethod
    def Hnh(H, Ls) -> Array:
        return H - 0.5j * jnp.sum(dag(Ls) @ Ls, axis=0)

    @property
    def Id(self) -> Array:
        return jnp.eye(self.H.shape[-1], dtype=self.H.dtype)
