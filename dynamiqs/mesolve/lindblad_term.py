import diffrax as dx
import jax.numpy as jnp
from jax.typing import Array, PyTree, Scalar

from ..time_array import TimeArray
from ..utils.utils import dag


class LindbladTerm(dx.ODETerm):
    H: TimeArray  # (n, n)
    Ls: TimeArray  # (nL, n, n)

    def vector_field(self, t: Scalar, rho: PyTree, _args: PyTree):
        Ls_t = self.Ls(t)
        out = -1j * self.Hnh(t) @ rho + 0.5 * jnp.sum(Ls_t @ rho @ dag(Ls_t), axis=0)
        return out + out.mH

    def Hnh(self, t: Scalar) -> Array:
        Ls_t = self.Ls(t)
        return self.H(t) - 0.5j * jnp.sum(dag(Ls_t) @ Ls_t, axis=0)

    @property
    def Id(self) -> Array:
        return jnp.eye(self.H.shape[-1], dtype=self.H.dtype)
