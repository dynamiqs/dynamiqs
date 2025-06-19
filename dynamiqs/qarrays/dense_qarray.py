from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, get_args

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, Device
from jaxtyping import ArrayLike
from qutip import Qobj

from .layout import Layout, dense
from .qarray import QArray, QArrayLike, in_last_two_dims, isqarraylike, to_jax
from .sparsedia_primitives import array_to_sparsedia

if TYPE_CHECKING:
    from .sparsedia_qarray import SparseDIAQArray

__all__ = ['DenseQArray']


class DenseQArray(QArray):
    r"""A dense qarray is a qarray that uses JAX arrays as data storage."""

    data: Array

    __qarray_matmul_priority__ = 0

    @property
    def dtype(self) -> jnp.dtype:
        return self.data.dtype

    @property
    def layout(self) -> Layout:
        return dense

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def mT(self) -> QArray:
        data = self.data.mT
        return replace(self, data=data)

    def conj(self) -> QArray:
        data = self.data.conj()
        return replace(self, data=data)

    def _reshape_unchecked(self, *shape: int) -> QArray:
        data = jnp.reshape(self.data, shape)
        return replace(self, data=data)

    def broadcast_to(self, *shape: int) -> QArray:
        data = jnp.broadcast_to(self.data, shape)
        return replace(self, data=data)

    def ptrace(self, *keep: int) -> QArray:
        from ..utils.general import ptrace  # noqa: PLC0415

        return ptrace(self.data, keep, self.dims)

    def powm(self, n: int) -> QArray:
        data = jnp.linalg.matrix_power(self.data, n)
        return replace(self, data=data)

    def expm(self, *, max_squarings: int = 16) -> QArray:
        data = jax.scipy.linalg.expm(self.data, max_squarings=max_squarings)
        return replace(self, data=data)

    def norm(self, *, psd: bool = True) -> Array:
        from ..utils.general import norm  # noqa: PLC0415

        return norm(self.data, psd=psd)

    def trace(self) -> Array:
        return self.data.trace(axis1=-1, axis2=-2)

    def sum(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        data = self.data.sum(axis=axis)

        # return array if last two dimensions are modified, qarray otherwise
        if in_last_two_dims(axis, self.ndim):
            return data
        else:
            return replace(self, data=data)

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        data = self.data.squeeze(axis=axis)

        # return array if last two dimensions are modified, qarray otherwise
        if in_last_two_dims(axis, self.ndim):
            return data
        else:
            return replace(self, data=data)

    def _eig(self) -> tuple[Array, QArray]:
        evals, evecs = jax.lax.linalg.eig(self.data, compute_left_eigenvectors=False)
        return evals, replace(self, data=evecs)

    def _eigh(self) -> tuple[Array, Array]:
        return jnp.linalg.eigh(self.data)

    def _eigvals(self) -> Array:
        return jnp.linalg.eigvals(self.data)

    def _eigvalsh(self) -> Array:
        return jnp.linalg.eigvalsh(self.data)

    def devices(self) -> set[Device]:
        return self.data.devices()

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return jnp.allclose(self.data, self.data.mT.conj(), rtol=rtol, atol=atol)

    def to_qutip(self) -> Qobj | list[Qobj]:
        return array_to_qobj_list(self.to_jax(), self.dims)

    def to_jax(self) -> Array:
        return self.data

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        return np.asarray(self.data, dtype=dtype)

    def asdense(self) -> DenseQArray:
        return self

    def assparsedia(self, offsets: tuple[int, ...] | None = None) -> SparseDIAQArray:
        from .sparsedia_qarray import SparseDIAQArray  # noqa: PLC0415

        offsets, diags = array_to_sparsedia(self.data, offsets)
        return SparseDIAQArray(self.dims, self.vectorized, offsets, diags)

    def block_until_ready(self) -> QArray:
        _ = self.data.block_until_ready()
        return self

    def __repr__(self) -> str:
        return super().__repr__() + f'\n{self.data}'

    def __mul__(self, y: ArrayLike) -> QArray:
        super().__mul__(y)

        data = y * self.data
        return replace(self, data=data)

    def __add__(self, y: QArrayLike) -> QArray:
        if isinstance(y, int | float) and y == 0:
            return self

        super().__add__(y)

        if isinstance(y, DenseQArray):
            data = self.data + y.data
        elif isinstance(y, get_args(ArrayLike)):
            data = self.data + to_jax(y)
        else:
            return NotImplemented

        return replace(self, data=data)

    def __matmul__(self, y: QArrayLike) -> QArray | Array:
        out = super().__matmul__(y)
        if out is NotImplemented:
            return NotImplemented

        if isinstance(y, DenseQArray):
            data = self.data @ y.data
        elif isqarraylike(y):
            data = self.data @ to_jax(y)
        else:
            return NotImplemented

        if self.isbra() and y.isket():
            return data

        return replace(self, data=data)

    def __rmatmul__(self, y: QArrayLike) -> QArray:
        super().__rmatmul__(y)

        if isinstance(y, DenseQArray):
            data = y.data @ self.data
        elif isqarraylike(y):
            data = to_jax(y) @ self.data
        else:
            return NotImplemented

        return replace(self, data=data)

    def __and__(self, y: QArray) -> QArray:
        super().__and__(y)

        if isinstance(y, DenseQArray):
            dims = self.dims + y.dims
            data = _bkron(self.data, y.data)
        else:
            return NotImplemented

        return replace(self, dims=dims, data=data)

    def addscalar(self, y: ArrayLike) -> QArray:
        data = self.data + to_jax(y)
        return replace(self, data=data)

    def elmul(self, y: QArrayLike) -> QArray:
        from .sparsedia_qarray import SparseDIAQArray  # noqa: PLC0415

        super().elmul(y)

        if isinstance(y, SparseDIAQArray):
            return y.elmul(self)

        data = self.data * to_jax(y)
        return replace(self, data=data)

    def elpow(self, power: int) -> QArray:
        data = self.data**power
        return replace(self, data=data)

    def __getitem__(self, key: int | slice) -> QArray:
        data = self.data[key]
        return replace(self, data=data)


def array_to_qobj_list(x: Array, dims: tuple[int, ...]) -> Qobj | list[Qobj]:
    # convert dims to qutip
    dims = list(dims)
    if x.shape[-1] == 1:  # [[3], [1]] or [[3, 4], [1, 1]]
        dims = [dims, [1] * len(dims)]
    elif x.shape[-2] == 1:  # [[1], [3]] or [[1, 1], [3, 4]]
        dims = [[1] * len(dims), dims]
    elif x.shape[-1] == x.shape[-2]:  # [[3], [3]] or [[3, 4], [3, 4]]
        dims = [dims, dims]

    return jax.tree.map(
        lambda x: Qobj(x, dims=dims),
        x.tolist(),
        is_leaf=lambda x: jnp.asarray(x).ndim == 2,
    )


@partial(jnp.vectorize, signature='(a,b),(c,d)->(ac,bd)')
def _bkron(a: Array, b: Array) -> Array:
    # batched kronecker product
    return jnp.kron(a, b)
