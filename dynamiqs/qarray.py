from __future__ import annotations

import warnings
from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import qutip as qt
from jax import Array
from jax.scipy.linalg import expm
from jaxtyping import ArrayLike, ScalarLike

from .utils.utils.general import (
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
    unit,
)

__all__ = ['QArray', 'DenseQArray']


# a decorator that takes a class method f and returns g(f(x))
def pack_dims(method: callable) -> callable:
    """Decorator to return a new QArray with the same dimensions as the original one."""

    def wrapper(self: QArray, *args, **kwargs) -> callable:
        return self.__class__(method(self, *args, **kwargs), dims=self.dims)

    return wrapper


class QArray(eqx.Module):
    r"""Quantum array object. DenseQArray is a wrapper around JAX arrays. It offers
    convenience methods to handle more easily quantum states
    (bras, kets and density matrices) and quantum operators.
    If you come from QuTiP, this is the equivalent of the Qobj class.
    """

    dims: tuple[int, ...]

    @property
    @abstractmethod
    def dtype(self) -> jnp.dtype:
        """Data type of the quantum state.

        Returns:
             The data type of the quantum state.
        """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of the quantum state.

        Returns:
            The shape of the quantum state.
        """

    @property
    def ndim(self) -> int:
        """Number of dimensions of the quantum state.

        Returns:
            The number of dimensions of the quantum state.
        """
        return len(self.dims)

    @property
    @abstractmethod
    def I(self) -> QArray:  # noqa: E743
        """Identity operator compatible with the quantum state.

        Returns:
            The identity operator.
        """

    @property
    @abstractmethod
    def mT(self) -> QArray:
        """Transpose of the quantum state.

        Returns:
            The transpose of the quantum state.
        """

    # methods
    @abstractmethod
    def conj(self) -> QArray:
        """Conjugate of the quantum state.

        Returns:
            The conjugate of the quantum state.
        """

    @abstractmethod
    def norm(self) -> Array:
        """Norm of the quantum state.

        Returns:
            The norm of the quantum state.
        """

    @abstractmethod
    def unit(self) -> QArray:
        """Normalize the quantum state.

        Returns:
            The normalized quantum state.
        """

    @abstractmethod
    def diag(self) -> Array:
        """Diagonal of the quantum state.

        Returns:
            The diagonal of the quantum state.
        """

    @abstractmethod
    def trace(self) -> Array:
        """Trace of the quantum state.

        Returns:
            The trace of the quantum state.
        """

    @abstractmethod
    def reshape(self, *new_shape: int) -> QArray:
        """Reshape the quantum state.

        Args:
            new_shape: New shape of the quantum state.

        Returns:
            The reshaped quantum state.
        """

    @abstractmethod
    def broadcast_to(self, *new_shape: int) -> QArray:
        """Broadcast the quantum state.

        Returns:
            The broadcast quantum state.
        """

    @abstractmethod
    def ptrace(self, keep_dims: tuple[int, ...]) -> QArray:
        """Partial trace of the quantum state.

        Args:
            keep_dims: Dimensions to keep.

        Returns:
            The partial trace of the quantum state.
        """

    @abstractmethod
    def dag(self) -> QArray:
        """Dagger of the quantum state.

        Returns:
            The dagger of the quantum state.
        """

    @abstractmethod
    def isket(self) -> bool:
        """Check if the quantum state is a ket.

        Returns:
            True if the quantum state is a ket, False otherwise.
        """

    @abstractmethod
    def isbra(self) -> bool:
        """Check if the quantum state is a bra.

        Returns:
            True if the quantum state is a bra, False otherwise.
        """

    @abstractmethod
    def isdm(self) -> bool:
        """Check if the quantum state is a density matrix.

        Returns:
            True if the quantum state is a density matrix, False otherwise.
        """

    def isop(self) -> bool:
        """Check if the quantum state is an operator.

        Returns:
            True if the quantum state is an operator, False otherwise.
        """
        return self.isdm()

    @abstractmethod
    def isherm(self) -> bool:
        """Check if the quantum state is Hermitian.

        Returns:
            True if the quantum state is Hermitian, False otherwise.
        """

    @abstractmethod
    def toket(self) -> QArray:
        """Convert the quantum state to a ket.

        Returns:
            The ket representation of the quantum state.
        """

    @abstractmethod
    def tobra(self) -> QArray:
        """Convert the quantum state to a bra.

        Returns:
            The bra representation of the quantum state.
        """

    @abstractmethod
    def todm(self) -> QArray:
        """Convert the quantum state to a density matrix.

        Returns:
            The density matrix representation of the quantum state.
        """

    def toop(self) -> QArray:
        """Convert the quantum state to an operator.

        Returns:
            The operator representation of the quantum state.
        """
        return self.todm()

    def proj(self) -> QArray:
        """Projector of the quantum state.

        Returns:
            The projector of the quantum state.
        """

    # conversion methods
    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """Convert the quantum state to a NumPy array.

        Returns:
            The NumPy array representation of the quantum state.
        """

    def to_qutip(self) -> qt.QObj:
        """Convert the quantum state to a QuTiP object.

        Returns:
            A QuTiP object representation of the quantum state.
        """
        return qt.Qobj(self.to_numpy(), dims=self.dims)

    @abstractmethod
    def to_jax(self) -> Array:
        """Convert the quantum state to a JAX array.

        Returns:
            The JAX array representation of the quantum state.
        """

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}(shape={self.shape}, '
            f'dims={self.dims}, dtype={self.dtype})'
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __neg__(self) -> QArray:
        """Negate the quantum state."""
        return -1 * self

    def __mul__(
        self, other: ScalarLike | ArrayLike @ abstractmethod
    ) -> QArray:  # warning if used with array
        """Element-wise multiplication with a scalar or an array."""

    def __rmul__(
        self, other: ScalarLike | ArrayLike
    ) -> QArray:  # warning if used with array
        """Element-wise multiplication with a scalar or an array on the right."""

    def __add__(
        self, other: ScalarLike | ArrayLike @ abstractmethod
    ) -> QArray:  # warning if used with scalar
        """Element-wise addition with a scalar or an array."""

    def __radd__(
        self, other: ScalarLike | ArrayLike
    ) -> QArray:  # warning if used with scalar
        """Element-wise addition with a scalar or an array on the right."""
        return self + other

    def __sub__(
        self, other: ScalarLike | ArrayLike
    ) -> QArray:  # warning if used with scalar
        """Element-wise subtraction with a scalar or an array."""

    def __rsub__(
        self, other: ScalarLike | ArrayLike
    ) -> QArray:  # warning if used with scalar
        """Element-wise subtraction with a scalar or an array on the right."""
        return -self + other

    @abstractmethod
    def __matmul__(self, other: ArrayLike) -> QArray:
        """Matrix multiplication with another quantum state or JAX array."""

    def __rmatmul__(self, other: ArrayLike) -> QArray:
        """Matrix multiplication with another quantum state or JAX array
        on the right.
        """


class DenseQArray(QArray):
    r"""DenseQArray is QArray that uses JAX arrays as data storage."""

    data: ArrayLike
    dims: tuple[int, ...]

    def __init__(self, data: ArrayLike, dims: tuple[int, ...] | None = None):
        if not (isbra(data) or isket(data) or isdm(data) or isop(data)):
            raise ValueError(
                f'DenseQArray data must be a bra, a ket, a density matrix '
                f'or and operator. Got array with size {data.shape}'
            )
        if dims is None:
            dims = data.shape[-2] if isket(data) else data.shape[-1]
            dims = (dims,)

        self.data = data
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

    @property
    @pack_dims
    def mT(self) -> Array:
        return self.data.mT

    @pack_dims
    def conj(self) -> Array:
        return self.data.conj()

    def norm(self) -> Array:
        return norm(self.data)

    @pack_dims
    def unit(self) -> Array:
        return unit(self.data)

    @pack_dims
    def diag(self) -> Array:
        return jnp.diag(self.data)

    def trace(self) -> Array:
        return jnp.trace(self.data)

    @pack_dims
    def reshape(self, *new_shape: int) -> Array:
        return jnp.reshape(self.data, new_shape)

    @pack_dims
    def broadcast_to(self, *new_shape: int) -> Array:
        return jnp.broadcast_to(self.data, new_shape)

    def ptrace(self, keep_dims: tuple[int, ...]) -> DenseQArray:
        return DenseQArray(
            ptrace(self.data, keep_dims, self.dims),
            tuple(self.dims[dim] for dim in keep_dims),
        )

    @pack_dims
    def dag(self) -> Array:
        """Dagger of the quantum state.

        Returns:
            DenseQArray: The dagger of the quantum state.
        """
        return dag(self.data)

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
                    f'Two DenseQArray must have the same dimensions to be added. '
                    f'Got {self.dims} and {other.dims}'
                )
            return DenseQArray(self.data + other.data, self.dims)
        elif isinstance(other, ScalarLike):
            warnings.warn(
                '"+" between a scalar and a DenseQArray performs '
                'element-wise addition. If you want to perform addition '
                'with an operator, use "x + 2 * x.I"',
                stacklevel=2,
            )
        return DenseQArray(self.data + other, self.dims)

    def __sub__(self, other: ScalarLike | ArrayLike | DenseQArray) -> DenseQArray:
        if isinstance(other, DenseQArray):
            if self.dims != other.dims:
                raise ValueError(
                    f'Two DenseQArray must have the same dimensions to be subtracted. '
                    f'Got {self.dims} and {other.dims}'
                )
            return DenseQArray(self.data - other.data, self.dims)
        elif isinstance(other, ScalarLike):
            warnings.warn(
                '"-" between a scalar and a DenseQArray performs '
                'element-wise addition. If you want to perform addition '
                'with an operator, use "x - 2 * x.I"',
                stacklevel=2,
            )
        return DenseQArray(self.data - other, self.dims)

    def __mul__(self, other: ScalarLike | ArrayLike | DenseQArray) -> DenseQArray:
        if isinstance(other, (ArrayLike, DenseQArray)):
            warnings.warn(
                '"*" between a DenseQArray and another DenseQArray '
                'or an Array performs element-wise multiplication. If you '
                'wanted to perform matrix multiplication, use "@" operator',
                stacklevel=2,
            )

        if isinstance(other, DenseQArray):
            if self.dims != other.dims:
                raise ValueError(
                    f'Two DenseQArray must have the same dimensions to be multiplied '
                    f'element wise. Got {self.dims} and {other.dims}'
                )
            return DenseQArray(self.data * other.data, self.dims)
        else:
            return DenseQArray(self.data * other, self.dims)

    def __matmul__(self, other: ScalarLike | ArrayLike | DenseQArray) -> DenseQArray:
        if isinstance(other, DenseQArray):
            if self.dims != other.dims:
                raise ValueError(
                    f'Two DenseQArray must have the same dimensions to be multiplied. '
                    f'Got {self.dims} and {other.dims}'
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
