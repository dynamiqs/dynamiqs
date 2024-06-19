from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import qutip as qt
from jax import Array
from jaxtyping import ArrayLike, ScalarLike

__all__ = ['QArray']


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
    def __mul__(self, y: ScalarLike | ArrayLike) -> QArray:
        """Element-wise multiplication with a scalar or an array."""
        # warning if used with array

    def __rmul__(self, y: ScalarLike | ArrayLike) -> QArray:
        """Element-wise multiplication with a scalar or an array on the right."""
        # warning if used with array
        return y * self

    @abstractmethod
    def __add__(self, y: ScalarLike | ArrayLike) -> QArray:
        """Element-wise addition with a scalar or an array."""
        # warning if used with scalar

    def __radd__(self, y: ScalarLike | ArrayLike) -> QArray:
        """Element-wise addition with a scalar or an array on the right."""
        # warning if used with scalar
        return self + y

    def __sub__(self, y: ScalarLike | ArrayLike) -> QArray:
        """Element-wise subtraction with a scalar or an array."""
        self + (-y)

    def __rsub__(self, y: ScalarLike | ArrayLike) -> QArray:
        """Element-wise subtraction with a scalar or an array on the right."""
        # warning if used with scalar
        return -self + y

    @abstractmethod
    def __matmul__(self, y: ArrayLike) -> QArray:
        """Matrix multiplication with another quantum state or JAX array."""

    @abstractmethod
    def __rmatmul__(self, y: ArrayLike) -> QArray:
        """Matrix multiplication with another quantum state or JAX array
        on the right.
        """

    @abstractmethod
    def __and__(self, y: QArray) -> QArray:
        """Tensor product between two quantum states."""

    @abstractmethod
    def __pow__(self, power: int) -> QArray:
        """Matrix power of the quantum state."""
