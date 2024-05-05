from __future__ import annotations

import warnings

import equinox as eqx
import jax.numpy as jnp
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
    ptrace,
    sinm,
    tensor,
    unit,
)

__all__ = ['QArray']


class QArray(eqx.Module):
    r"""Quantum array object. Qarray is a wrapper around JAX arrays. It offers
    convenience methods to handle more easily quantum states
    (bras, kets and density matrices) and quantum operators.
    If you come from QuTiP, this is the equivalent of the Qobj class.
    """

    inner: ArrayLike
    dims: tuple[int, ...]

    def __init__(self, inner: ArrayLike, dims: tuple[int, ...] | None = None):
        if not (isbra(inner) or isket(inner) or isdm(inner) or isop(inner)):
            raise ValueError(
                f'QArray data must be a bra, a ket, a density matrix '
                f'or and operator. Got array with size {inner.shape}'
            )
        if dims is None:
            dims = inner.shape[-2] if isket(inner) else inner.shape[-1]
            dims = (dims,)

        self.inner = inner
        self.dims = dims

    @property
    def I(self) -> Array:  # noqa: E743
        """Identity operator compatible with the quantum state.

        Returns:
            The identity operator.
        """
        return jnp.eye(jnp.prod(jnp.asarray(self.dims)))

    def dag(self) -> QArray:
        """Dagger of the quantum state.

        Returns:
            QArray: The dagger of the quantum state.
        """
        return QArray(dag(self.inner), self.dims)

    def ptrace(self, keep_dims: tuple[int, ...]) -> QArray:
        """Partial trace of the quantum state.

        Args:
            keep_dims: tuple[int, ...]
                Dimensions to keep in the partial trace.

        Returns:
            QArray: The partial trace of the quantum state.
        """
        return QArray(
            ptrace(self.inner, keep_dims, self.dims),
            tuple(self.dims[dim] for dim in keep_dims),
        )

    def expm(self) -> QArray:
        """Matrix exponential of the quantum state.

        Returns:
            QArray: The matrix exponential of the quantum state.
        """
        return QArray(expm(self.inner), self.dims)

    def cosm(self) -> QArray:
        """Matrix cosine of the quantum state.

        Returns:
            QArray: The matrix cosine of the quantum state.
        """
        return QArray(cosm(self.inner), self.dims)

    def sinm(self) -> QArray:
        """Matrix sine of the quantum state.

        Returns:
            QArray: The matrix sine of the quantum state.
        """
        return QArray(sinm(self.inner), self.dims)

    def norm(self) -> Array:
        """Norm of the quantum state.

        Returns:
            float: The norm of the quantum state.
        """
        return norm(self.inner)

    def unit(self) -> QArray:
        """Normalize the quantum state.

        Returns:
            QArray: The normalized quantum state.
        """
        return QArray(unit(self.inner), self.dims)

    def __add__(self, other: ScalarLike | ArrayLike | QArray) -> QArray:
        if isinstance(other, QArray):
            if self.dims != other.dims:
                raise ValueError(
                    f'Two QArray must have the same dimensions to be added. '
                    f'Got {self.dims} and {other.dims}'
                )
            return QArray(self.inner + other.inner, self.dims)
        elif isinstance(other, ScalarLike):
            warnings.warn(
                '"+" between a scalar and a QArray performs element-wise addition. '
                'If you want to perform addition with an operator, use "x + 2 * x.I"',
                stacklevel=2,
            )
        return QArray(self.inner + other, self.dims)

    def __sub__(self, other: ScalarLike | ArrayLike | QArray) -> QArray:
        if isinstance(other, QArray):
            if self.dims != other.dims:
                raise ValueError(
                    f'Two QArray must have the same dimensions to be subtracted. '
                    f'Got {self.dims} and {other.dims}'
                )
            return QArray(self.inner - other.inner, self.dims)
        elif isinstance(other, ScalarLike):
            warnings.warn(
                '"-" between a scalar and a QArray performs element-wise addition. '
                'If you want to perform addition with an operator, use "x - 2 * x.I"',
                stacklevel=2,
            )
        return QArray(self.inner - other, self.dims)

    def __mul__(self, other: ScalarLike | ArrayLike | QArray) -> QArray:
        if isinstance(other, (ArrayLike, QArray)):
            warnings.warn(
                '"*" between a QArray and another QArray or an Array performs '
                'element-wise multiplication. If you wanted to perform matrix '
                'multiplication, use "@" operator',
                stacklevel=2,
            )

        if isinstance(other, QArray):
            if self.dims != other.dims:
                raise ValueError(
                    f'Two QArray must have the same dimensions to be multiplied '
                    f'element wise. Got {self.dims} and {other.dims}'
                )
            return QArray(self.inner * other.inner, self.dims)
        else:
            return QArray(self.inner * other, self.dims)

    def __matmul__(self, other: ScalarLike | ArrayLike | QArray) -> QArray:
        if isinstance(other, QArray):
            if self.dims != other.dims:
                raise ValueError(
                    f'Two QArray must have the same dimensions to be multiplied. '
                    f'Got {self.dims} and {other.dims}'
                )
            return QArray(self.inner @ other.inner, self.dims)
        else:
            return QArray(self.inner @ other, self.dims)

    def __rmatmul__(self, other: ScalarLike | ArrayLike | QArray) -> QArray:
        if isinstance(other, QArray):
            if self.dims != other.dims:
                raise ValueError(
                    f'Two QArray must have the same dimensions to be multiplied. '
                    f'Got {self.dims} and {other.dims}'
                )
            return QArray(other.inner @ self.inner, self.dims)
        else:
            return QArray(other @ self.inner, self.dims)

    def __and__(self, other: QArray) -> QArray:
        """Tensor product between two quantum states."""
        return QArray(tensor(self.inner, other.inner), self.dims + other.dims)
