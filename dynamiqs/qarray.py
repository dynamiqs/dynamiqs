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
        """Return the data type of the quantum state.

        Returns:
             The data type of the quantum state.
        """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the quantum state.

        Returns:
            The shape of the quantum state.
        """

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the quantum state.

        Returns:
            The number of dimensions of the quantum state.
        """
        return len(self.dims)

    @property
    @abstractmethod
    def I(self) -> QArray:  # noqa: E743
        """Returns the identity operator compatible with the quantum state.

        Returns:
            The identity operator.
        """

    # methods
    @abstractmethod
    def conj(self) -> QArray:
        """Returns the conjugate of the quantum state.

        Returns:
            The conjugate of the quantum state.
        """

    @abstractmethod
    def dag(self) -> QArray:
        """Returns the dagger of the quantum state.

        Returns:
            The dagger of the quantum state.
        """

    @abstractmethod
    def norm(self) -> Array:
        """Returns the norm of the quantum state.

        Returns:
            The norm of the quantum state.
        """

    def unit(self) -> QArray:
        """Returns the normalized the quantum state.

        Returns:
            The normalized quantum state.
        """
        return self / self.norm()[..., None, None]

    @abstractmethod
    def reshape(self, *new_shape: int) -> QArray:
        """Returns the reshape the quantum state.

        Args:
            new_shape: New shape of the quantum state.

        Returns:
            The reshaped quantum state.
        """

    @abstractmethod
    def broadcast_to(self, *new_shape: int) -> QArray:
        """Returns the broadcast the quantum state.

        Returns:
            The broadcast quantum state.
        """

    @abstractmethod
    def ptrace(self, keep: tuple[int, ...]) -> QArray:
        """Returns the partial trace of the quantum state.

        Args:
            keep: Dimensions to keep.

        Returns:
            The partial trace of the quantum state.
        """

    @abstractmethod
    def isket(self) -> bool:
        """Returns the check if the quantum state is a ket.

        Returns:
            True if the quantum state is a ket, False otherwise.
        """

    @abstractmethod
    def isbra(self) -> bool:
        """Returns the check if the quantum state is a bra.

        Returns:
            True if the quantum state is a bra, False otherwise.
        """

    @abstractmethod
    def isdm(self) -> bool:
        """Returns the check if the quantum state is a density matrix.

        Returns:
            True if the quantum state is a density matrix, False otherwise.
        """

    def isop(self) -> bool:
        """Returns the check if the quantum state is an operator.

        Returns:
            True if the quantum state is an operator, False otherwise.
        """
        return self.isdm()

    @abstractmethod
    def isherm(self) -> bool:
        """Returns the check if the quantum state is Hermitian.

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
        return self.todm()

    # conversion methods
    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """Convert the quantum state to a NumPy array.

        Returns:
            The NumPy array representation of the quantum state.
        """

    @abstractmethod
    def to_qutip(self) -> qt.QObj:
        """Convert the quantum state to a QuTiP object.

        Returns:
            A QuTiP object representation of the quantum state.
        """

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

    def __neg__(self) -> QArray:
        """Negate the quantum state."""
        return -1 * self

    @abstractmethod
    def __mul__(
        self, other: ScalarLike | ArrayLike
    ) -> QArray:  # warning if used with array
        """Element-wise multiplication with a scalar or an array."""

    def __rmul__(
        self, other: ScalarLike | ArrayLike
    ) -> QArray:  # warning if used with array
        """Element-wise multiplication with a scalar or an array on the right."""
        return other * self

    @abstractmethod
    def __add__(
        self, other: ScalarLike | ArrayLike @ abstractmethod
    ) -> QArray:  # warning if used with scalar
        """Element-wise addition with a scalar or an array."""

    @abstractmethod
    def __radd__(
        self, other: ScalarLike | ArrayLike
    ) -> QArray:  # warning if used with scalar
        """Element-wise addition with a scalar or an array on the right."""
        return self + other

    @abstractmethod
    def __sub__(
        self, other: ScalarLike | ArrayLike
    ) -> QArray:  # warning if used with scalar
        """Element-wise subtraction with a scalar or an array."""

    @abstractmethod
    def __rsub__(
        self, other: ScalarLike | ArrayLike
    ) -> QArray:  # warning if used with scalar
        """Element-wise subtraction with a scalar or an array on the right."""
        return -self + other

    @abstractmethod
    def __matmul__(self, other: ArrayLike) -> QArray:
        """Matrix multiplication with another quantum state or JAX array."""

    @abstractmethod
    def __rmatmul__(self, other: ArrayLike) -> QArray:
        """Matrix multiplication with another quantum state or JAX array
        on the right.
        """

    @abstractmethod
    def __and__(self, other: QArray) -> QArray:
        """Tensor product between two quantum states."""

    @abstractmethod
    def __pow__(self, power: int) -> QArray:
        """Matrix power of the quantum state."""


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
            ptrace(self.data, keep, self.dims),
            tuple(self.dims[dim] for dim in keep),
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
