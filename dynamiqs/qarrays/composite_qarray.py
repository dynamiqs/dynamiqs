from __future__ import annotations

from math import prod

import jax.numpy as jnp
import numpy as np
from jax import Array, Device
from jaxtyping import ArrayLike
from qutip import Qobj

from .layout import Layout
from .qarray import QArray

__all__ = ['CompositeQArray']

from .dense_qarray import DenseQArray
from .qarray import QArrayLike
from .sparsedia_qarray import SparseDIAQArray


class CompositeQArray(QArray):
    terms: list[tuple[QArray, ...]]
    """List of tuple of `QArray`s. Each element of the list is called a _term_. Each
    term is composed of _factors_."""

    @property
    def _first_term(self) -> tuple[QArray, ...]:
        """Return the first term of the sum, raise if there's none."""
        try:
            return self.terms[0]
        except IndexError as e:
            raise ValueError('The CompositeQArray is empty.') from e

    @property
    def _first_factor(self) -> QArray:
        """Return the first factor of the first term of the sum, raise if there's
        none.
        """
        try:
            return self._first_term[0]
        except IndexError as e:
            raise ValueError('The first term of the CompositeQArray is empty') from e

    @property
    def dtype(self) -> jnp.dtype:
        return self._first_factor.dtype

    @property
    def layout(self) -> Layout:
        return NotImplemented  # TODO

    @property
    def shape(self) -> tuple[int, ...]:
        bshape = ()  # TODO: find broadcasted batching shape with jnp.broadcast_shapes
        ns = [factor.shape[-1] for factor in self._first_term]
        ntot = prod(ns)
        return (*bshape, ntot, ntot)

    @property
    def mT(self) -> QArray:
        terms = [tuple(factor.mT for factor in term) for term in self.terms]
        return CompositeQArray(self.dims, self.vectorized, terms)

    def conj(self) -> QArray:
        terms = [tuple(factor.conj() for factor in term) for term in self.terms]
        return CompositeQArray(self.dims, self.vectorized, terms)

    def _reshape_unchecked(self, *shape: int) -> QArray:
        # TODO
        pass

    def broadcast_to(self, *shape: int) -> QArray:
        # TODO
        pass

    def ptrace(self, *keep: int) -> QArray:
        pass

    def powm(self, n: int) -> QArray:
        pass

    def expm(self, *, max_squarings: int = 16) -> QArray:
        pass

    def norm(self) -> Array:
        pass

    def trace(self) -> Array:
        pass

    def sum(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        pass

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        pass

    def _eig(self) -> tuple[Array, QArray]:
        pass

    def _eigh(self) -> tuple[Array, Array]:
        pass

    def _eigvals(self) -> Array:
        pass

    def _eigvalsh(self) -> Array:
        pass

    def devices(self) -> set[Device]:
        pass

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        pass

    def to_qutip(self) -> Qobj | list[Qobj]:
        pass

    def to_jax(self) -> Array:
        pass

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        pass

    def asdense(self) -> DenseQArray:
        pass

    def assparsedia(self) -> SparseDIAQArray:
        pass

    def block_until_ready(self) -> QArray:
        pass

    def __mul__(self, y: ArrayLike) -> QArray:
        pass

    def __add__(self, y: QArrayLike) -> QArray:
        pass

    def __matmul__(self, y: QArrayLike) -> QArray | Array:
        pass

    def __rmatmul__(self, y: QArrayLike) -> QArray:
        pass

    def __and__(self, y: QArray) -> QArray:
        pass

    def addscalar(self, y: ArrayLike) -> QArray:
        pass

    def elmul(self, y: QArrayLike) -> QArray:
        pass

    def elpow(self, power: int) -> QArray:
        pass

    def __getitem__(self, key: int | slice) -> QArray:
        pass
