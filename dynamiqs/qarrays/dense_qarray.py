from __future__ import annotations

from typing import get_args

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike
from qutip import Qobj

from ..utils.jax_utils import to_qutip
from ..utils.utils.general import (
    dag,
    expm,
    isbra,
    isdm,
    isherm,
    isket,
    isop,
    norm,
    powm,
    ptrace,
    tensor,
    tobra,
    todm,
    toket,
    trace,
)
from .qarray import QArray

__all__ = ['dense_qarray', 'DenseQArray']


def dense_qarray(data: ArrayLike, dims: tuple[int, ...] | None = None) -> DenseQArray:
    if not (isbra(data) or isket(data) or isdm(data) or isop(data)):
        raise ValueError(
            f'DenseQArray data must be a bra, a ket, a density matrix '
            f'or and operator. Got array with size {data.shape}'
        )
    if dims is None:
        dims = data.shape[-2] if isket(data) else data.shape[-1]
        dims = (dims,)

    data = jnp.asarray(data)
    return DenseQArray(dims, data)


class DenseQArray(QArray):
    r"""DenseQArray is QArray that uses JAX arrays as data storage."""

    data: jax.Array

    @property
    def dtype(self) -> jnp.dtype:
        return self.data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def mT(self) -> QArray:
        data = self.data.mT
        return DenseQArray(self.dims, data)

    @property
    def I(self) -> QArray:  # noqa: E743
        data = jnp.eye(jnp.prod(jnp.asarray(self.dims)))
        return DenseQArray(self.dims, data)

    def conj(self) -> QArray:
        data = self.data.conj()
        return DenseQArray(self.dims, data)

    def dag(self) -> QArray:
        data = dag(self.data)
        return DenseQArray(self.dims, data)

    def trace(self) -> jax.Array:
        return trace(self.data)

    def sum(self, axis: int | tuple[int, ...] | None = None) -> jax.Array:
        return self.data.sum(axis=axis)

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> jax.Array:
        return self.data.squeeze(axis=axis)

    def norm(self) -> jax.Array:
        return norm(self.data)

    def reshape(self, *shape: int) -> QArray:
        data = jnp.reshape(self.data, shape)
        return DenseQArray(self.dims, data)

    def broadcast_to(self, *shape: int) -> QArray:
        data = jnp.broadcast_to(self.data, shape)
        return DenseQArray(self.dims, data)

    def ptrace(self, keep: tuple[int, ...]) -> QArray:
        dims = tuple(self.dims[dim] for dim in keep)
        data = ptrace(self.data, keep, self.dims)
        return DenseQArray(dims, data)

    def isket(self) -> bool:
        return isket(self.data)

    def isbra(self) -> bool:
        return isbra(self.data)

    def isdm(self) -> bool:
        return isdm(self.data)

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return isherm(self.data, rtol=rtol, atol=atol)

    def toket(self) -> QArray:
        data = toket(self.data)
        return DenseQArray(self.dims, data)

    def tobra(self) -> QArray:
        data = tobra(self.data)
        return DenseQArray(self.dims, data)

    def todm(self) -> QArray:
        data = todm(self.data)
        return DenseQArray(self.dims, data)

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.data)

    def to_qutip(self) -> Qobj:
        return to_qutip(self.data, dims=self.dims)

    def to_jax(self) -> jax.Array:
        return self.data

    def _powm(self, n: int) -> QArray:
        data = powm(self.data, n)
        return DenseQArray(self.dims, data)

    def _expm(self, n: int) -> QArray:
        data = expm(self.data, n)
        return DenseQArray(self.dims, data)

    def __mul__(self, y: ArrayLike) -> QArray:
        super().__mul__(y)

        if isinstance(y, DenseQArray):
            data = self.data * y.data
        elif isinstance(y, get_args(ArrayLike)):
            data = self.data * y
        else:
            return NotImplemented

        return DenseQArray(self.dims, data)

    def __add__(self, y: ArrayLike) -> QArray:
        super().__add__(y)

        if isinstance(y, DenseQArray):
            data = self.data + y.data
        elif isinstance(y, get_args(ArrayLike)):
            data = self.data + y
        else:
            return NotImplemented

        return DenseQArray(self.dims, data)

    def __matmul__(self, y: ArrayLike) -> QArray:
        super().__matmul__(y)

        if isinstance(y, DenseQArray):
            data = self.data @ y.data
        elif isinstance(y, get_args(ArrayLike)):
            data = self.data @ y
        else:
            return NotImplemented

        return DenseQArray(self.dims, data)

    def __rmatmul__(self, y: ArrayLike) -> QArray:
        super().__rmatmul__(y)

        if isinstance(y, DenseQArray):
            data = y.data @ self.data
        elif isinstance(y, get_args(ArrayLike)):
            data = y @ self.data
        else:
            return NotImplemented

        return DenseQArray(self.dims, data)

    def __and__(self, y: QArray) -> QArray:
        super().__and__(y)
        dims = self.dims + y.dims

        if isinstance(y, DenseQArray):
            data = tensor(self.data, y.data)
        else:
            return NotImplemented

        return DenseQArray(dims, data)

    def __pow__(self, power: int) -> QArray:
        super().__pow__(power)
        data = powm(self.data, power)
        return DenseQArray(self.dims, data)
