from __future__ import annotations

from typing import get_args

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, Device
from jaxtyping import ArrayLike
from qutip import Qobj

from .._utils import _is_batched_scalar
from .qarray import QArray, _in_last_two_dims
from .types import QArrayLike, asjaxarray, isqarraylike

__all__ = ['DenseQArray']

# batched Kronecker product of two arrays
_bkron = jnp.vectorize(jnp.kron, signature='(a,b),(c,d)->(ac,bd)')


class DenseQArray(QArray):
    r"""DenseQArray is QArray that uses JAX arrays as data storage."""

    data: Array

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

    def conj(self) -> QArray:
        data = self.data.conj()
        return DenseQArray(self.dims, data)

    def reshape(self, *shape: int) -> QArray:
        data = jnp.reshape(self.data, shape)
        return DenseQArray(self.dims, data)

    def broadcast_to(self, *shape: int) -> QArray:
        data = jnp.broadcast_to(self.data, shape)
        return DenseQArray(self.dims, data)

    def ptrace(self, keep: tuple[int, ...]) -> QArray:
        from ..utils.quantum_utils.general import ptrace

        dims = tuple(self.dims[dim] for dim in keep)
        data = ptrace(self.data, keep, self.dims)
        return DenseQArray(dims, data)

    def powm(self, n: int) -> QArray:
        data = jnp.linalg.matrix_power(self.data, n)
        return DenseQArray(self.dims, data)

    def expm(self, *, max_squarings: int = 16) -> QArray:
        data = jax.scipy.linalg.expm(self.data, max_squarings=max_squarings)
        return DenseQArray(self.dims, data)

    def norm(self) -> Array:
        from ..utils.quantum_utils.general import norm

        return norm(self.data)

    def trace(self) -> Array:
        return self.data.trace(axis1=-1, axis2=-2)

    def sum(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        data = self.data.sum(axis=axis)

        # return array if last two dimensions are modified, qarray otherwise
        if _in_last_two_dims(axis, self.ndim):
            return data
        else:
            return DenseQArray(self.dims, data)

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        data = self.data.squeeze(axis=axis)

        # return array if last two dimensions are modified, qarray otherwise
        if _in_last_two_dims(axis, self.ndim):
            return data
        else:
            return DenseQArray(self.dims, data)

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
        from .utils import to_qutip

        return to_qutip(self.data, dims=self.dims)

    def to_jax(self) -> Array:
        return self.data

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        return np.asarray(self.data, dtype=dtype)

    def __repr__(self) -> str:
        return super().__repr__() + f', layout=dense\n{self.data}'

    def __mul__(self, y: QArrayLike) -> QArray:
        super().__mul__(y)

        if _is_batched_scalar(y):
            data = self.data * y
        elif isinstance(y, DenseQArray):
            data = self.data * y.data
        elif isqarraylike(y):
            data = self.data * asjaxarray(y)
        else:
            return NotImplemented

        return DenseQArray(self.dims, data)

    def __truediv__(self, y: QArrayLike) -> QArray:
        super().__truediv__(y)

        if _is_batched_scalar(y):
            data = self.data / y
        elif isinstance(y, DenseQArray):
            data = self.data / y.data
        elif isqarraylike(y):
            data = self.data / asjaxarray(y)
        else:
            return NotImplemented

        return DenseQArray(self.dims, data)

    def __add__(self, y: QArray) -> QArray:
        from .sparse_dia_qarray import SparseDIAQArray

        super().__add__(y)

        if _is_batched_scalar(y):
            data = self.data + y
        elif isinstance(y, DenseQArray):
            data = self.data + y.data
        elif isinstance(y, get_args(ArrayLike)):
            data = self.data + asjaxarray(y)
        elif isinstance(y, SparseDIAQArray):
            return y.__add__(self)
        else:
            return NotImplemented

        return DenseQArray(self.dims, data)

    def __matmul__(self, y: QArrayLike) -> QArray | Array:
        super().__matmul__(y)

        if isinstance(y, DenseQArray):
            dims = self.dims if len(self.dims) < len(y.dims) else y.dims
            data = self.data @ y.data
        elif isqarraylike(y):
            dims = self.dims
            data = self.data @ asjaxarray(y)
        else:
            return NotImplemented

        if self.isbra() and y.isket():
            return data

        return DenseQArray(dims, data)

    def __rmatmul__(self, y: QArrayLike) -> QArray:
        super().__rmatmul__(y)

        if isinstance(y, DenseQArray):
            dims = self.dims if len(self.dims) < len(y.dims) else y.dims
            data = y.data @ self.data
        elif isqarraylike(y):
            dims = self.dims
            data = asjaxarray(y) @ self.data
        else:
            return NotImplemented

        return DenseQArray(dims, data)

    def __and__(self, y: QArray) -> QArray:
        super().__and__(y)
        dims = self.dims + y.dims

        if isinstance(y, DenseQArray):
            data = _bkron(self.data, y.data)
        else:
            return NotImplemented

        return DenseQArray(dims, data)

    def _pow(self, power: int) -> QArray:
        data = self.data**power
        return DenseQArray(self.dims, data)

    def __getitem__(self, key: int | slice) -> QArray:
        data = self.data[key]
        return DenseQArray(self.dims, data)
