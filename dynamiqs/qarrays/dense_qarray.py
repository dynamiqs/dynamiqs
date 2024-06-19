from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import qutip as qt
from jax import Array
from jax.scipy.linalg import expm
from jaxtyping import ArrayLike, ScalarLike

from ..utils.utils.general import (
    cosm,
    dag,
    isbra,
    isdm,
    isket,
    isop,
    norm,
    proj,
    ptrace,
    sinm,
    tensor,
    tobra,
    todm,
    toket,
)
from .qarray import QArray, pack_dims

__all__ = ['DenseQArray']


class DenseQArray(QArray):
    r"""DenseQArray is QArray that uses JAX arrays as data storage."""

    data: Array

    def __init__(self, data: ArrayLike, dims: tuple[int, ...] | None = None):
        if not (isbra(data) or isket(data) or isdm(data) or isop(data)):
            raise ValueError(
                f'DenseQArray data must be a bra, a ket, a density matrix '
                f'or and operator. Got array with size {data.shape}'
            )
        if dims is None:
            dims = data.shape[-2] if isket(data) else data.shape[-1]
            dims = (dims,)

        self.data = jnp.asarray(data)
        self.dims = dims

    @property
    def dtype(self) -> jnp.dtype:
        return self.data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    @pack_dims
    def I(self) -> Array:  # noqa: E743
        return jnp.eye(jnp.prod(jnp.asarray(self.dims)))

    @pack_dims
    def conj(self) -> Array:
        return self.data.conj()

    @pack_dims
    def dag(self) -> Array:
        """Dagger of the quantum state.

        Returns:
            DenseQArray: The dagger of the quantum state.
        """
        return dag(self.data)

    def norm(self) -> Array:
        return norm(self.data)

    @pack_dims
    def reshape(self, *new_shape: int) -> Array:
        return jnp.reshape(self.data, new_shape)

    @pack_dims
    def broadcast_to(self, *new_shape: int) -> Array:
        return jnp.broadcast_to(self.data, new_shape)

    def ptrace(self, keep: tuple[int, ...]) -> DenseQArray:
        return DenseQArray(
            ptrace(self.data, keep, self.dims), tuple(self.dims[dim] for dim in keep)
        )

    def isket(self) -> bool:
        return isket(self.data)

    def isbra(self) -> bool:
        return isbra(self.data)

    def isdm(self) -> bool:
        return isdm(self.data)

    def isherm(self) -> bool:
        # TODO: could be made more efficient
        return jnp.allclose(self.data, dag(self.data))

    @pack_dims
    def toket(self) -> Array:
        return toket(self.data)

    @pack_dims
    def tobra(self) -> Array:
        return tobra(self.data)

    @pack_dims
    def todm(self) -> Array:
        return todm(self.data)

    @pack_dims
    def proj(self) -> Array:
        return proj(self.data)

    def to_numpy(self) -> np.ndarray:
        return np.array(self.data)

    def to_jax(self) -> Array:
        return self.data

    def to_qutip(self) -> qt.QObj:
        return qt.Qobj(self.to_numpy(), dims=self.dims)

    @pack_dims
    def expm(self) -> Array:
        """Matrix exponential of the quantum state.

        Returns:
            DenseQArray: The matrix exponential of the quantum state.
        """
        return expm(self.data)

    @pack_dims
    def cosm(self) -> Array:
        """Matrix cosine of the quantum state.

        Returns:
            DenseQArray: The matrix cosine of the quantum state.
        """
        return cosm(self.data)

    @pack_dims
    def sinm(self) -> Array:
        """Matrix sine of the quantum state.

        Returns:
            DenseQArray: The matrix sine of the quantum state.
        """
        return sinm(self.data)

    def __add__(self, other: ScalarLike | ArrayLike | DenseQArray) -> DenseQArray:
        if isinstance(other, DenseQArray):
            if self.dims != other.dims:
                raise ValueError(
                    f'QArrays must have the same dimensions to be added. '
                    f'Got {self.dims} and {other.dims}'
                )
            return DenseQArray(self.data + other.data, self.dims)
        elif isinstance(other, ScalarLike):
            warnings.warn(
                'Calling `+` between a scalar and a QArray performs '
                'element-wise addition. If you want to perform addition '
                'with the identity operator, use `x + 2 * x.I` instead.',
                stacklevel=2,
            )
        return DenseQArray(self.data + other, self.dims)

    def __sub__(self, other: ScalarLike | ArrayLike | DenseQArray) -> DenseQArray:
        if isinstance(other, DenseQArray):
            if self.dims != other.dims:
                raise ValueError(
                    f'QArrays must have the same dimensions to be subtracted '
                    f'from one another. Got {self.dims} and {other.dims}.'
                )
            return DenseQArray(self.data - other.data, self.dims)
        elif isinstance(other, ScalarLike):
            warnings.warn(
                'Calling `-` between a scalar and a QArray performs '
                'element-wise subtraction. If you want to perform subtraction '
                'with the identity operator, use `x - 2 * x.I` instead.',
                stacklevel=2,
            )
        return DenseQArray(self.data - other, self.dims)

    def __mul__(self, other: ScalarLike | ArrayLike | DenseQArray) -> DenseQArray:
        if isinstance(other, (ArrayLike, DenseQArray)):
            warnings.warn(
                'Calling `*` between a QArray and another QArray or Array '
                'performs element-wise multiplication. If you want to perform '
                'matrix multiplication, use the `@` operator instead.',
                stacklevel=2,
            )

        if isinstance(other, DenseQArray):
            if self.dims != other.dims:
                raise ValueError(
                    f'QArrays must have the same dimensions to be multiplied '
                    f'together element-wise. Got {self.dims} and {other.dims}.'
                )
            return DenseQArray(self.data * other.data, self.dims)
        else:
            return DenseQArray(self.data * other, self.dims)

    def __matmul__(self, other: ScalarLike | ArrayLike | DenseQArray) -> DenseQArray:
        if isinstance(other, DenseQArray):
            if self.dims != other.dims:
                raise ValueError(
                    f'QArrays must have the same dimensions to be multiplied '
                    f'together. Got {self.dims} and {other.dims}.'
                )
            return DenseQArray(self.data @ other.data, self.dims)
        else:
            return DenseQArray(self.data @ other, self.dims)

    def __rmatmul__(self, other: ScalarLike | ArrayLike | DenseQArray) -> DenseQArray:
        if isinstance(other, DenseQArray):
            if self.dims != other.dims:
                raise ValueError(
                    f'Two DenseQArray must have the same dimensions to be multiplied. '
                    f'Got {self.dims} and {other.dims}'
                )
            return DenseQArray(other.data @ self.data, self.dims)
        else:
            return DenseQArray(other @ self.data, self.dims)

    def __and__(self, other: DenseQArray) -> DenseQArray:
        """Tensor product between two quantum states."""
        return DenseQArray(tensor(self.data, other.data), self.dims + other.dims)

    def __pow__(self, power: int) -> DenseQArray:
        warnings.warn(
            'Calling `**` between on a QArray performs '
            'element-wise power. If you want to perform matrix power, '
            'use `x @ x @ ... @ x` instead.',
            stacklevel=2,
        )
