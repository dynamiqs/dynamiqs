from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, get_args

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, Device
from jaxtyping import ArrayLike
from qutip import Qobj

from .data_array import DataArray
from .layout import Layout, dense
from .sparsedia_primitives import array_to_sparsedia

if TYPE_CHECKING:
    from .sparsedia_qarray import SparseDIAArray

__all__ = ['DenseArray']


class DenseArray(DataArray):
    r"""A dense array is a data array that uses JAX arrays as data storage."""

    array: Array

    __dataarray_matmul_priority__ = 0

    def _replace(
        self,
        array: Array | None = None,
    ) -> DenseArray:
        if array is None:
            array = self.array
        return super()._replace(array=array)

    @property
    def dtype(self) -> jnp.dtype:
        return self.array.dtype

    @property
    def layout(self) -> Layout:
        return dense

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def mT(self) -> DenseArray:
        array = self.array.mT
        return self._replace(array=array)

    def conj(self) -> DenseArray:
        array = self.array.conj()
        return self._replace(array=array)

    def _reshape_unchecked(self, *shape: int) -> DenseArray:
        array = jnp.reshape(self.array, shape)
        return self._replace(array=array)

    def broadcast_to(self, *shape: int) -> DenseArray:
        array = jnp.broadcast_to(self.array, shape)
        return self._replace(array=array)

    def powm(self, n: int) -> DenseArray:
        array = jnp.linalg.matrix_power(self.array, n)
        return self._replace(array=array)

    def expm(self, *, max_squarings: int = 16) -> DenseArray:
        array = jax.scipy.linalg.expm(self.array, max_squarings=max_squarings)
        return self._replace(array=array)

    def trace(self) -> Array:
        return self.array.trace(axis1=-1, axis2=-2)

    def sum(self, axis: int | tuple[int, ...] | None = None) -> DenseArray | Array:
        array = self.array.sum(axis=axis)
        return self._replace(array=array)

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> DenseArray | Array:
        array = self.array.squeeze(axis=axis)
        return self._replace(array=array)

    def devices(self) -> set[Device]:
        return self.array.devices()

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return jnp.allclose(self.array, self.array.mT.conj(), rtol=rtol, atol=atol)

    def to_qutip(self) -> Qobj | list[Qobj]:
        return array_to_qobj_list(self.to_jax(), self.dims)

    def to_jax(self) -> Array:
        return self.array

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        return np.asarray(self.array, dtype=dtype)

    def asdense(self) -> DenseArray:
        return self

    def assparsedia(self) -> SparseDIAArray:
        from .sparsedia_array import SparseDIAArray

        offsets, diags = array_to_sparsedia(self.array)
        return SparseDIAArray(self.dims, self.vectorized, offsets, diags)

    def block_until_ready(self) -> DenseArray:
        _ = self.array.block_until_ready()
        return self

    def __repr__(self) -> str:
        return super().__repr__() + f'\n{self.array}'

    def __mul__(self, y: ArrayLike) -> DenseArray:
        super().__mul__(y)

        array = y * self.array
        return self._replace(array=array)

    def __add__(self, y: ArrayLike) -> DenseArray:
        if isinstance(y, int | float) and y == 0:
            return self

        super().__add__(y)

        if isinstance(y, DenseArray):
            array = self.array + y.array
        elif isinstance(y, get_args(ArrayLike)):
            array = self.array + to_jax(y)
        else:
            return NotImplemented

        return self._replace(array=array)

    def __matmul__(self, y: ArrayLike) -> DenseArray:
        super().__matmul__(y)
        if isinstance(y, DenseArray):
            array = self.array @ y.array
        elif isqarraylike(y):
            array = self.array @ to_jax(y)
        else:
            return NotImplemented

        if self.isbra() and y.isket():
            return array

        return self._replace(array=array)

    def __rmatmul__(self, y: ArrayLike) -> DenseArray:
        super().__rmatmul__(y)

        if isinstance(y, DenseArray):
            array = y.array @ self.array
        elif isqarraylike(y):
            array = to_jax(y) @ self.array
        else:
            return NotImplemented

        return self._replace(array=array)

    def __and__(self, y: ArrayLike) -> DenseArray:
        super().__and__(y)

        if isinstance(y, DenseArray):
            dims = self.dims + y.dims
            array = _bkron(self.array, y.array)
        else:
            return NotImplemented

        return self._replace(dims=dims, array=array)

    def __getitem__(self, key: int | slice) -> DenseArray:
        array = self.array[key]
        return self._replace(array=array)


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
