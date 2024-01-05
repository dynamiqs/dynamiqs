import diffrax as dx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree, Scalar

from ..utils.utils import dag


class LindbladTerm(dx.AbstractTerm):
    H: Array  # (n, n)
    Ls: Array  # (nL, n, n)

    def vf(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        pass

    @staticmethod
    def contr(t0: Scalar, t1: Scalar) -> Scalar:
        return t1 - t0

    @staticmethod
    def prod(vf: PyTree, control: Scalar) -> PyTree:
        return jtu.tree_map(lambda v: control * v, vf)

    @property
    def Hnh(self) -> Array:
        return self.H - 0.5j * jnp.sum(dag(self.Ls) @ self.Ls, axis=0)

    @property
    def Id(self) -> Array:
        return jnp.eye(self.H.shape[-1], dtype=self.H.dtype)
