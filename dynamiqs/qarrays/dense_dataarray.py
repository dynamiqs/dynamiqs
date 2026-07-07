from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, ClassVar, get_args

import jax
import jax.numpy as jnp
import numpy as np
from equinox.internal._omega import _Metaω  # noqa: PLC2403
from jax import Array, Device
from jaxtyping import ArrayLike
from qutip import Qobj

from .dataarray import (
    DataArray,
    DataArrayLike,
    IndexType,
    in_last_two_dims,
    key_touches_last_two_dims,
)
from .layout import Layout, dense
from .sparsedia_primitives import array_to_sparsedia

if TYPE_CHECKING:
    from .sparsedia_dataarray import SparseDIADataArray


class DenseDataArray(DataArray):
    r"""A dense data array using JAX arrays as data storage."""

    data: Array

    _matmul_priority: ClassVar[int] = 0

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
    def mT(self) -> DataArray:
        data = self.data.mT
        return replace(self, data=data)

    def conj(self) -> DataArray:
        data = self.data.conj()
        return replace(self, data=data)

    def _reshape_unchecked(self, *shape: int) -> DataArray:
        data = jnp.reshape(self.data, shape)
        return replace(self, data=data)

    def broadcast_to(self, *shape: int) -> DataArray:
        data = jnp.broadcast_to(self.data, shape)
        return replace(self, data=data)

    def _swapaxes_unchecked(self, axis1: int, axis2: int) -> DataArray:
        data = jnp.swapaxes(self.data, axis1, axis2)
        return replace(self, data=data)

    def _moveaxis_unchecked(
        self, source: tuple[int, ...], destination: tuple[int, ...]
    ) -> DataArray:
        data = jnp.moveaxis(self.data, source, destination)
        return replace(self, data=data)

    def _expand_dims_unchecked(self, axis: tuple[int, ...]) -> DataArray:
        data = jnp.expand_dims(self.data, axis=axis)
        return replace(self, data=data)

    def powm(self, n: int) -> DataArray:
        data = jnp.linalg.matrix_power(self.data, n)
        return replace(self, data=data)

    def expm(self, *, max_squarings: int = 16) -> DataArray:
        data = jax.scipy.linalg.expm(self.data, max_squarings=max_squarings)
        return replace(self, data=data)

    def norm(self, *, psd: bool = False) -> Array:
        from ..utils.general import norm  # noqa: PLC0415

        return norm(self.data, psd=psd)

    def trace(self) -> Array:
        return self.data.trace(axis1=-1, axis2=-2)

    def sum(self, axis: int | tuple[int, ...] | None = None) -> DataArray | Array:
        data = self.data.sum(axis=axis)

        # return array if last two dimensions are modified, DataArray otherwise
        if in_last_two_dims(axis, self.ndim):
            return data
        else:
            return replace(self, data=data)

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> DataArray | Array:
        data = self.data.squeeze(axis=axis)

        # return array if last two dimensions are modified, DataArray otherwise
        if in_last_two_dims(axis, self.ndim):
            return data
        else:
            return replace(self, data=data)

    def _eig(self) -> tuple[Array, DataArray]:
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

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> Array[bool]:
        return jnp.allclose(self.data, self.data.mT.conj(), rtol=rtol, atol=atol)

    def to_jax(self) -> Array:
        return self.data

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        return np.asarray(self.data, dtype=dtype)

    def asdense(self) -> DenseDataArray:
        return self

    def assparsedia(self, offsets: tuple[int, ...] | None = None) -> SparseDIADataArray:
        from .sparsedia_dataarray import SparseDIADataArray  # noqa: PLC0415

        offsets, diags = array_to_sparsedia(self.data, offsets)
        return SparseDIADataArray(offsets, diags)

    def block_until_ready(self) -> DataArray:
        _ = self.data.block_until_ready()
        return self

    def _repr_extra(self) -> str:
        return f'\n{self.data}'

    def __mul__(self, y: DataArrayLike) -> DataArray:
        from .sparsedia_dataarray import SparseDIADataArray  # noqa: PLC0415

        if isinstance(y, SparseDIADataArray):
            return y * self

        if isinstance(y, DenseDataArray):
            data = self.data * y.data
        elif isinstance(y, get_args(ArrayLike)):
            data = self.data * jnp.asarray(y)
        else:
            return NotImplemented

        return replace(self, data=data)

    def __add__(self, y: DataArrayLike) -> DataArray:
        if isinstance(y, int | float) and y == 0:
            return self

        if isinstance(y, DenseDataArray):
            data = self.data + y.data
        elif isinstance(y, get_args(ArrayLike)):
            data = self.data + jnp.asarray(y)
        else:
            return NotImplemented

        return replace(self, data=data)

    def __matmul__(self, y: DataArrayLike) -> DataArray | Array:
        if isinstance(y, DataArray) and self._matmul_priority < y._matmul_priority:
            return NotImplemented

        if isinstance(y, DenseDataArray):
            data = self.data @ y.data
        elif isinstance(y, get_args(ArrayLike)):
            data = self.data @ jnp.asarray(y)
        else:
            return NotImplemented

        return replace(self, data=data)

    def __rmatmul__(self, y: DataArrayLike) -> DataArray:
        if isinstance(y, DenseDataArray):
            data = y.data @ self.data
        elif isinstance(y, get_args(ArrayLike)):
            data = jnp.asarray(y) @ self.data
        else:
            return NotImplemented

        return replace(self, data=data)

    def __and__(self, y: DataArray) -> DataArray:
        if isinstance(y, DenseDataArray):
            data = _bkron(self.data, y.data)
        else:
            return NotImplemented

        return replace(self, data=data)

    def __pow__(self, power: int | _Metaω) -> DataArray:
        # to deal with the x**ω notation from equinox (used in diffrax internals)
        if isinstance(power, _Metaω):
            return _Metaω.__rpow__(power, self)

        data = self.data**power
        return replace(self, data=data)

    def __getitem__(self, key: IndexType) -> DataArray | Array:
        if key_touches_last_two_dims(key, self.ndim):
            return self.data[key]
        return replace(self, data=self.data[key])


def array_to_qobj_list(x: Array, dims: tuple[int, ...]) -> Qobj | list[Qobj]:
    # convert dims to qutip
    qobj_dims = list(dims)
    if x.shape[-1] == 1:  # [[3], [1]] or [[3, 4], [1, 1]]
        qobj_dims = [qobj_dims, [1] * len(qobj_dims)]
    elif x.shape[-2] == 1:  # [[1], [3]] or [[1, 1], [3, 4]]
        qobj_dims = [[1] * len(qobj_dims), qobj_dims]
    elif x.shape[-1] == x.shape[-2]:  # [[3], [3]] or [[3, 4], [3, 4]]
        qobj_dims = [qobj_dims, qobj_dims]

    return jax.tree.map(
        lambda x: Qobj(x, dims=qobj_dims),
        x.tolist(),
        is_leaf=lambda x: jnp.asarray(x).ndim == 2,
    )


@partial(jnp.vectorize, signature='(a,b),(c,d)->(ac,bd)')
def _bkron(a: Array, b: Array) -> Array:
    # batched kronecker product
    return jnp.kron(a, b)
