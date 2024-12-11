from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, get_args

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, Device
from jaxtyping import ArrayLike
from qutip import Qobj

from .._utils import _is_batched_scalar
from .layout import Layout, dense
from .qarray import QArray, QArrayLike, _in_last_two_dims, _to_jax, isqarraylike
from .sparsedia_primitives import array_to_sparsedia

if TYPE_CHECKING:
    from .sparse_dia_qarray import SparseDIAQArray

__all__ = ['DenseQArray']


class DenseQArray(QArray):
    r"""DenseQArray is QArray that uses JAX arrays as data storage."""

    data: Array

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
        return DenseQArray(self.dims, data)

    @property
    def _data(self) -> Array:
        return self.data

    def conj(self) -> QArray:
        data = self.data.conj()
        return DenseQArray(self.dims, data)

    def reshape(self, *shape: int) -> QArray:
        data = jnp.reshape(self.data, shape)
        return DenseQArray(self.dims, data)

    def broadcast_to(self, *shape: int) -> QArray:
        data = jnp.broadcast_to(self.data, shape)
        return DenseQArray(self.dims, data)

    def ptrace(self, *keep: int) -> QArray:
        from ..utils.general import ptrace

        return ptrace(self.data, keep, self.dims)

    def powm(self, n: int) -> QArray:
        data = jnp.linalg.matrix_power(self.data, n)
        return DenseQArray(self.dims, data)

    def expm(self, *, max_squarings: int = 16) -> QArray:
        data = jax.scipy.linalg.expm(self.data, max_squarings=max_squarings)
        return DenseQArray(self.dims, data)

    def norm(self) -> Array:
        from ..utils.general import norm

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

    def eig(self) -> tuple[Array, QArray]:
        evals, evecs = jax.lax.linalg.eig(self.data, compute_left_eigenvectors=False)
        return evals, DenseQArray(self.dims, evecs)

    def _eigh(self) -> tuple[Array, Array]:
        return jnp.linalg.eigh(self.data)

    def _eigvals(self) -> Array:
        return jnp.linalg.eigvals(self.data)

    def _eigvalsh(self) -> Array:
        return jnp.linalg.eigvalsh(self.data)

    def devices(self) -> set[Device]:
        return self.data.devices()

    def asdense(self) -> DenseQArray:
        return self

    def assparsedia(self) -> SparseDIAQArray:
        from .sparsedia_qarray import SparseDIAQArray

        offsets, diags = array_to_sparsedia(self.data)
        return SparseDIAQArray(self.dims, offsets, diags)

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return jnp.allclose(self.data, self.data.mT.conj(), rtol=rtol, atol=atol)

    def to_qutip(self) -> Qobj | list[Qobj]:
        return _array_to_qobj_list(self.to_jax(), self.dims)

    def to_jax(self) -> Array:
        return self.data

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        return np.asarray(self.data, dtype=dtype)

    def block_until_ready(self) -> QArray:
        _ = self.data.block_until_ready()
        return self

    def __repr__(self) -> str:
        return super().__repr__() + f'\n{self.data}'

    def __mul__(self, y: QArrayLike) -> QArray:
        super().__mul__(y)

        if _is_batched_scalar(y):
            data = self.data * y
        elif isinstance(y, DenseQArray):
            data = self.data * y.data
        elif isqarraylike(y):
            data = self.data * _to_jax(y)
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
            data = self.data / _to_jax(y)
        else:
            return NotImplemented

        return DenseQArray(self.dims, data)

    def __add__(self, y: QArray) -> QArray:
        super().__add__(y)

        if _is_batched_scalar(y):
            data = self.data + y
        elif isinstance(y, DenseQArray):
            data = self.data + y.data
        elif isinstance(y, get_args(ArrayLike)):
            data = self.data + _to_jax(y)
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
            data = self.data @ _to_jax(y)
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
            data = _to_jax(y) @ self.data
        else:
            return NotImplemented

        return DenseQArray(dims, data)

    def __and__(self, y: QArray) -> QArray:
        super().__and__(y)

        if isinstance(y, DenseQArray):
            dims = self.dims + y.dims
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


def _array_to_qobj_list(x: Array, dims: tuple[int, ...]) -> Qobj | list[Qobj]:
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
