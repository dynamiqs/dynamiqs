from __future__ import annotations

import warnings
from dataclasses import replace
from math import prod
from typing import get_args

import jax.numpy as jnp
import numpy as np
from jax import Array, Device
from jaxtyping import ArrayLike
from qutip import Qobj

from .dense_qarray import DenseQArray
from .layout import Layout
from .qarray import QArray, QArrayLike
from .sparsedia_qarray import SparseDIAQArray

__all__ = ['CompositeQArray']


class CompositeQArray(QArray):
    terms: list[tuple[QArray, ...]]
    """List of tuple of `QArray`s. Each element of the list is called a _term_. Each
    term is composed of _factors_."""

    def __check_init__(self):
        # check that there is at least one term
        if not self.terms:
            raise ValueError('CompositeQArray must have at least one term.')

        # check that each term has at least one factor
        for term in self.terms:
            if not term:
                raise ValueError(
                    'Each term in a CompositeQArray must have at least one factor.'
                )

        # check that all factors have the same dtype and devices
        for factor in self._all_factors():
            if factor.dtype != self.dtype:
                raise ValueError(
                    'All factors in a CompositeQArray must have the same dtype.'
                )
            if factor.devices() != self.devices():
                raise ValueError(
                    'All factors in a CompositeQArray must be on the same device.'
                )

        # check that all factors have broadcastable shapes
        try:
            jnp.broadcast_shapes(*[factor.shape[:-2] for factor in self._all_factors()])
        except ValueError as e:
            raise ValueError(
                'All factors in a CompositeQArray must have broadcastable shapes.'
            ) from e

    @property
    def _first_term(self) -> tuple[QArray, ...]:
        """Return the first term."""
        return self.terms[0]

    @property
    def _first_factor(self) -> QArray:
        """Return the first factor of the first term."""
        return self._first_term[0]

    def _all_factors(self) -> list[QArray]:
        """Return a list of all factors in all terms."""
        return [factor for term in self.terms for factor in term]

    @property
    def dtype(self) -> jnp.dtype:
        return self._first_factor.dtype

    @property
    def layout(self) -> Layout:
        return NotImplemented  # TODO

    @property
    def shape(self) -> tuple[int, ...]:
        bshape = jnp.broadcast_shapes(
            *[factor.shape[:-2] for factor in self._all_factors()]
        )
        ns = [factor.shape[-1] for factor in self._first_term]
        ntot = prod(ns)
        return *bshape, ntot, ntot

    @property
    def mT(self) -> QArray:
        terms = [tuple(factor.mT for factor in term) for term in self.terms]
        return replace(self, terms=terms)

    @property
    def nterms(self) -> int:
        return len(self.terms)

    def conj(self) -> QArray:
        terms = [tuple(factor.conj() for factor in term) for term in self.terms]
        return replace(self, terms=terms)

    def _reshape_unchecked(self, *shape: int) -> QArray:
        # TODO (skip for now)
        pass

    def broadcast_to(self, *shape: int) -> QArray:
        # TODO (skip for now)
        pass

    def ptrace(self, *keep: int) -> QArray:
        super().ptrace(*keep)

        dont_keep = [i for i in range(len(self.dims)) if i not in keep]

        def ptrace_term(term: tuple[QArray, ...]) -> tuple[QArray, ...]:
            traced_factors = prod(term[i].trace() for i in dont_keep)
            kept_factors = [term[i] for i in keep]
            kept_factors[0] *= traced_factors
            return tuple(kept_factors)

        terms = [ptrace_term(term) for term in self.terms]

        new_dims = tuple(self.dims[i] for i in keep)
        return replace(self, dims=new_dims, terms=terms)

    def powm(self, n: int) -> QArray:
        warnings.warn(
            'A `CompositeQArray` has been converted to a `DenseQArray` while attempting'
            ' to compute its matrix power.',
            stacklevel=2,
        )
        return self.asdense().powm(n=n)

    def expm(self, *, max_squarings: int = 16) -> QArray:
        warnings.warn(
            'A `CompositeQArray` has been converted to a `DenseQArray` while attempting'
            ' to compute its matrix exponential.',
            stacklevel=2,
        )
        return self.asdense().expm(max_squarings=max_squarings)

    def norm(self) -> Array:
        return self.trace().real

    def trace(self) -> Array:
        return sum(prod(factor.trace() for factor in term) for term in self.terms)

    def sum(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        # TODO (skip for now)
        pass

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        # TODO (skip for now)
        pass

    def _eig(self) -> tuple[Array, QArray]:
        warnings.warn(
            'A `CompositeQArray` has been converted to a `DenseQArray` while attempting'
            ' to compute its eigen-decomposition.',
            stacklevel=2,
        )
        return self.asdense()._eig()

    def _eigh(self) -> tuple[Array, Array]:
        warnings.warn(
            'A `CompositeQArray` has been converted to a `DenseQArray` while attempting'
            ' to compute its eigen-decomposition.',
            stacklevel=2,
        )
        return self.asdense()._eigh()

    def _eigvals(self) -> Array:
        warnings.warn(
            'A `CompositeQArray` has been converted to a `DenseQArray` while attempting'
            ' to compute its eigen-decomposition.',
            stacklevel=2,
        )
        return self.asdense()._eigvals()

    def _eigvalsh(self) -> Array:
        warnings.warn(
            'A `CompositeQArray` has been converted to a `DenseQArray` while attempting'
            ' to compute its eigen-decomposition.',
            stacklevel=2,
        )
        return self.asdense()._eigvalsh()

    def devices(self) -> set[Device]:
        return self._first_factor.devices()

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return all(factor.isherm(rtol, atol) for factor in self._all_factors())

    def to_qutip(self) -> Qobj | list[Qobj]:
        return self.asdense().to_qutip()

    def to_jax(self) -> Array:
        return self.asdense().to_jax()

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        return self.asdense().__array__(dtype=dtype, copy=copy)

    def asdense(self) -> DenseQArray:
        from ..utils.general import tensor

        return sum(
            tensor(*(factor.asdense() for factor in term)) for term in self.terms
        )

    def assparsedia(self) -> SparseDIAQArray:
        from ..utils.general import tensor

        return sum(
            tensor(*(factor.assparsedia() for factor in term)) for term in self.terms
        )

    def block_until_ready(self) -> QArray:
        _ = self._first_factor.block_until_ready()
        return self

    def __repr__(self) -> str:
        return (
            f'QArray: shape={self.shape}, dims={self.dims}, dtype={self.dtype}, '
            f'type=composite, nterms={self.nterms}'
        )

    def __mul__(self, y: ArrayLike) -> QArray:
        super().__mul__(y)

        terms = [(y * term[0], *term[1:]) for term in self.terms]
        return replace(self, terms=terms)

    def __add__(self, y: QArrayLike) -> QArray:
        if isinstance(y, int | float) and y == 0:
            return self

        super().__add__(y)

        if isinstance(y, CompositeQArray):
            return replace(self, terms=self.terms + y.terms)
        elif isinstance(y, DenseQArray | SparseDIAQArray):
            return replace(self, terms=[*self.terms, (y,)])
        elif isinstance(y, get_args(ArrayLike)):
            warnings.warn(
                'A CompositeQArray has been converted to a DenseQArray due to'
                ' element-wise addition with an ArrayLike.',
                stacklevel=2,
            )
            return self.asdense() + y

        return NotImplemented

    def __matmul__(self, y: QArrayLike) -> QArray | Array:
        # TODO: not sure about the implementation
        out = super().__matmul__(y)
        if out is NotImplemented:
            return NotImplemented

        warnings.warn(
            'A `CompositeQArray` has been converted to a `DenseQArray` while attempting'
            ' to multiply with a matrix.',
            stacklevel=2,
        )
        return self.asdense() @ y

    def __rmatmul__(self, y: QArrayLike) -> QArray:
        super().__rmatmul__(y)

        warnings.warn(
            'A `CompositeQArray` has been converted to a `DenseQArray` while attempting'
            ' to multiply with a matrix.',
            stacklevel=2,
        )
        return y @ self.asdense()

    def __and__(self, y: QArray) -> QArray:
        super().__and__(y)

        if isinstance(y, CompositeQArray):
            terms = [terms_a + terms_b for terms_a in self.terms for terms_b in y.terms]
            return replace(self, dims=self.dims + y.dims, terms=terms)

        dims = self.dims + y.dims
        terms = [(*term, y) for term in self.terms]
        return replace(self, dims=dims, terms=terms)

    def addscalar(self, y: ArrayLike) -> QArray:
        warnings.warn(
            'A `CompositeQArray` has been converted to a `DenseQArray` while attempting'
            ' to adding a scalar.',
            stacklevel=2,
        )
        return self.asdense().addscalar(y)

    def elmul(self, y: QArrayLike) -> QArray:
        # TODO: not sure about the implementation
        super().elmul(y)

        warnings.warn(
            'A `CompositeQArray` has been converted to a `DenseQArray` while attempting'
            ' to compute its element-wise multiplication.',
            stacklevel=2,
        )
        return self.asdense().elmul(y)

    def elpow(self, power: int) -> QArray:
        warnings.warn(
            'A `CompositeQArray` has been converted to a `DenseQArray` while attempting'
            ' to compute its element-wise power.',
            stacklevel=2,
        )
        return self.asdense().elpow(power)

    def __getitem__(self, key: int | slice) -> QArray:
        # TODO (skip for now)
        pass
