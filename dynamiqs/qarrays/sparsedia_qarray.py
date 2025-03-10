from __future__ import annotations

import re
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from qutip import Qobj

from .dense_qarray import DenseQArray
from .layout import Layout, dia
from .qarray import (
    QArray,
    QArrayLike,
    _in_last_two_dims,
    _include_last_two_dims,
    _is_key_in_batch_dims,
    _to_jax,
    isqarraylike,
)
from .sparsedia_primitives import (
    _sparsedia_slice,
    add_sparsedia_sparsedia,
    and_sparsedia_sparsedia,
    broadcast_sparsedia,
    matmul_array_sparsedia,
    matmul_sparsedia_array,
    matmul_sparsedia_sparsedia,
    mul_sparsedia_array,
    mul_sparsedia_sparsedia,
    powm_sparsedia,
    reshape_sparsedia,
    shape_sparsedia,
    sparsedia_to_array,
    trace_sparsedia,
    transpose_sparsedia,
)

__all__ = ['SparseDIAQArray']


class SparseDIAQArray(QArray):
    offsets: tuple[int, ...] = eqx.field(static=True)
    diags: Array = eqx.field(converter=jnp.asarray)

    def _replace(
        self,
        dims: tuple[int, ...] | None = None,
        vectorized: bool | None = None,
        offsets: tuple[int, ...] | None = None,
        diags: Array | None = None,
    ) -> SparseDIAQArray:
        if offsets is None:
            offsets = self.offsets
        if diags is None:
            diags = self.diags
        return super()._replace(
            dims=dims, vectorized=vectorized, offsets=offsets, diags=diags
        )

    def __check_init__(self):
        # check diags and offsets have the right type and shape before compressing them
        if not isinstance(self.offsets, tuple):
            raise TypeError(
                'Argument `offsets` of `SparseDIAQArray` must be a tuple but got '
                f'{type(self.offsets)}'
            )

        if self.diags.ndim < 2 or self.diags.shape[-2] != len(self.offsets):
            raise ValueError(
                'Argument `diags` of `SparseDIAQArray` must be of shape '
                f'(..., len(offsets), prod(dims)), but got {self.diags.shape}'
            )

        # check that diagonals contain zeros outside the bounds of the matrix using
        # equinox runtime checks
        error = (
            'Diagonals of a `SparseDIAQArray` must contain zeros outside the '
            'matrix bounds.'
        )
        for i, offset in enumerate(self.offsets):
            zero_slice = slice(None, offset) if offset >= 0 else slice(offset, None)
            check = self.diags[..., i, zero_slice] != 0
            eqx.error_if(self.diags, check, error)

    @property
    def dtype(self) -> jnp.dtype:
        return self.diags.dtype

    @property
    def layout(self) -> Layout:
        return dia

    @property
    def shape(self) -> tuple[int, ...]:
        return shape_sparsedia(self.diags)

    @property
    def mT(self) -> QArray:
        offsets, diags = transpose_sparsedia(self.offsets, self.diags)
        return self._replace(offsets=offsets, diags=diags)

    @property
    def ndiags(self) -> int:
        return len(self.offsets)

    def conj(self) -> QArray:
        diags = self.diags.conj()
        return self._replace(diags=diags)

    def _reshape_unchecked(self, *shape: int) -> QArray:
        offsets, diags = reshape_sparsedia(self.offsets, self.diags, shape)
        return self._replace(offsets=offsets, diags=diags)

    def broadcast_to(self, *shape: int) -> QArray:
        if shape[-2:] != self.shape[-2:]:
            raise ValueError(
                f'Cannot broadcast to shape {shape} because the last two dimensions do '
                f'not match current shape dimensions, {self.shape}.'
            )

        offsets, diags = broadcast_sparsedia(self.offsets, self.diags, shape)
        return self._replace(offsets=offsets, diags=diags)

    def ptrace(self, *keep: int) -> QArray:
        raise NotImplementedError

    def powm(self, n: int) -> QArray:
        offsets, diags = powm_sparsedia(self.offsets, self.diags, n)
        return self._replace(offsets=offsets, diags=diags)

    def expm(self, *, max_squarings: int = 16) -> QArray:
        warnings.warn(
            'A `SparseDIAQArray` has been converted to a `DenseQArray` while computing '
            'its matrix exponential.',
            stacklevel=2,
        )
        x = sparsedia_to_array(self.offsets, self.diags)
        expm_x = jax.scipy.linalg.expm(x, max_squarings=max_squarings)
        return DenseQArray(self.dims, self.vectorized, expm_x)

    def norm(self) -> Array:
        return self.trace()

    def trace(self) -> Array:
        return trace_sparsedia(self.offsets, self.diags)

    def sum(self, axis: int | tuple[int, ...] | None = None) -> Array:
        # return array if last two dimensions are modified, qarray otherwise
        if _in_last_two_dims(axis, self.ndim):
            if _include_last_two_dims(axis, self.ndim):
                return self.diags.sum(axis)
            else:
                return self.to_jax().sum(axis)
        else:
            diags = self.diags.sum(axis)
            return self._replace(diags=diags)

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        # return array if last two dimensions are modified, qarray otherwise
        if _in_last_two_dims(axis, self.ndim):
            if _include_last_two_dims(axis, self.ndim):
                return self.diags.squeeze(axis)
            else:
                return self.to_jax().squeeze(axis)
        else:
            diags = self.diags.squeeze(axis)
            return self._replace(diags=diags)

    def _eig(self) -> tuple[Array, QArray]:
        warnings.warn(
            'A `SparseDIAQArray` has been converted to a `DenseQArray` while attempting'
            ' to compute its eigen-decomposition.',
            stacklevel=2,
        )
        return self.asdense()._eig()

    def _eigh(self) -> tuple[Array, Array]:
        warnings.warn(
            'A `SparseDIAQArray` has been converted to a `DenseQArray` while attempting'
            ' to compute its eigen-decomposition.',
            stacklevel=2,
        )
        return self.asdense()._eigh()

    def _eigvals(self) -> Array:
        warnings.warn(
            'A `SparseDIAQArray` has been converted to a `DenseQArray` while attempting'
            ' to compute its eigen-decomposition.',
            stacklevel=2,
        )
        return self.asdense()._eigvals()

    def _eigvalsh(self) -> Array:
        warnings.warn(
            'A `SparseDIAQArray` has been converted to a `DenseQArray` while attempting'
            ' to compute its eigen-decomposition.',
            stacklevel=2,
        )
        return self.asdense()._eigvalsh()

    def devices(self) -> set[jax.Device]:
        raise NotImplementedError

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        # TODO: Improve this by using a direct QArray comparison function, once it is
        # implemented. This will avoid materalizing the dense matrix.
        return self.asdense().isherm(rtol=rtol, atol=atol)

    def to_qutip(self) -> Qobj | list[Qobj]:
        return self.asdense().to_qutip()

    def to_jax(self) -> Array:
        return self.asdense().to_jax()

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        return self.asdense().__array__(dtype=dtype, copy=copy)

    def asdense(self) -> DenseQArray:
        data = sparsedia_to_array(self.offsets, self.diags)
        return DenseQArray(self.dims, self.vectorized, data)

    def assparsedia(self) -> SparseDIAQArray:
        return self

    def block_until_ready(self) -> QArray:
        _ = self.diags.block_until_ready()
        return self

    def __repr__(self) -> str:
        # === array representation with dots instead of zeros
        if jnp.issubdtype(self.dtype, jnp.complexfloating):
            # match '0. +0.j' with any number of spaces
            pattern = r'(?<!\d)0\.\s*(\+|\-)0\.j'
        elif jnp.issubdtype(self.dtype, jnp.floating):
            # match '0.' with any number of spaces
            pattern = r'(?<!\d)0\.\s*'
        elif jnp.issubdtype(self.dtype, jnp.integer):
            # match '0' with any number of spaces
            pattern = r'(?<!\d)0\s*'
        else:
            raise ValueError(
                'Unsupported dtype for `SparseDIAQArray` representation, got '
                f'{self.dtype}.'
            )

        # replace with a centered dot of the same length as the matched string
        replace_with_dot = lambda match: f'{"⋅":^{len(match.group(0))}}'
        data_str = re.sub(pattern, replace_with_dot, str(self.to_jax()))
        return super().__repr__() + f', ndiags={self.ndiags}\n{data_str}'

    def __mul__(self, y: ArrayLike) -> QArray:
        super().__mul__(y)

        diags = y * self.diags
        return self._replace(diags=diags)

    def __add__(self, y: QArrayLike) -> QArray:
        if isinstance(y, int | float) and y == 0:
            return self

        super().__add__(y)

        if isinstance(y, SparseDIAQArray):
            offsets, diags = add_sparsedia_sparsedia(
                self.offsets, self.diags, y.offsets, y.diags
            )
            return self._replace(offsets=offsets, diags=diags)
        elif isqarraylike(y):
            warnings.warn(
                'A sparse qarray has been converted to dense layout due to element-wise'
                ' addition with a dense qarray.',
                stacklevel=2,
            )
            return self.asdense() + y

        return NotImplemented

    def __matmul__(self, y: QArrayLike) -> QArray:
        super().__matmul__(y)

        if isinstance(y, SparseDIAQArray):
            offsets, diags = matmul_sparsedia_sparsedia(
                self.offsets, self.diags, y.offsets, y.diags
            )
            return self._replace(offsets=offsets, diags=diags)
        elif isqarraylike(y):
            y = _to_jax(y)
            data = matmul_sparsedia_array(self.offsets, self.diags, y)
            return DenseQArray(self.dims, self.vectorized, data)

        return NotImplemented

    def __rmatmul__(self, y: QArrayLike) -> QArray:
        super().__rmatmul__(y)

        if isqarraylike(y):
            y = _to_jax(y)
            data = matmul_array_sparsedia(y, self.offsets, self.diags)
            return DenseQArray(self.dims, self.vectorized, data)

        return NotImplemented

    def __and__(self, y: QArray) -> QArray:
        if isinstance(y, SparseDIAQArray):
            offsets, diags = and_sparsedia_sparsedia(
                self.offsets, self.diags, y.offsets, y.diags
            )
            dims = self.dims + y.dims
            return self._replace(dims=dims, offsets=offsets, diags=diags)
        elif isinstance(y, DenseQArray):
            return self.asdense() & y

        return NotImplemented

    def __rand__(self, y: QArray) -> QArray:
        if isinstance(y, DenseQArray):
            return y & self.asdense()

        return NotImplemented

    def addscalar(self, y: ArrayLike) -> QArray:
        warnings.warn(
            'A sparse qarray has been converted to dense layout due to element-wise '
            'addition with a scalar.',
            stacklevel=2,
        )
        return self.asdense().addscalar(y)

    def elmul(self, y: QArrayLike) -> QArray:
        super().elmul(y)

        if isinstance(y, SparseDIAQArray):
            offsets, diags = mul_sparsedia_sparsedia(
                self.offsets, self.diags, y.offsets, y.diags
            )
        else:
            offsets, diags = mul_sparsedia_array(self.offsets, self.diags, _to_jax(y))

        return self._replace(offsets=offsets, diags=diags)

    def elpow(self, power: int) -> QArray:
        diags = self.diags**power
        return self._replace(diags=diags)

    def __getitem__(self, key: int | slice | tuple) -> QArray | jax.Array:
        if key in (slice(None, None, None), Ellipsis):
            return self

        if _is_key_in_batch_dims(key, self.ndim):
            diags = self.diags[key]
            return self._replace(diags=diags)
        else:
            # todo: handle out of bounds indexes
            # return a jax qarray that materializes the sparse array
            full_slice = slice(None, None, None)
            if Ellipsis in key:
                ellipsis_key = key.index(Ellipsis)
                key = (
                    key[:ellipsis_key]
                    + (full_slice,) * (self.ndim - len(key) + 1)
                    + key[ellipsis_key + 1 :]
                )

            normalized_key = key
            # for normalized key to scribe each dimension
            while len(normalized_key) < len(self.shape):
                normalized_key.append(slice(None, None, None))

            normalized_key = normalized_key[:-2] + tuple(
                [
                    normalize_slice(s, k)
                    for (s, k) in zip(
                        normalized_key[-2:], self.shape[-2:], strict=False
                    )
                ]
            )
            normalized_batch_key = normalized_key[:-2]

            result = jnp.vectorize(
                materialize_single_matrix,
                signature='(a,b)->(c,d)',
                excluded=('offsets', 'key', 'normalized_key', 'dtype'),
            )(
                self.diags[normalized_batch_key],
                offsets=self.offsets,
                normalized_key=normalized_key,
                dtype=self.dtype,
            )

            # since we converted integer indexes on the non batching dimension
            # to slices, we have to squeeze them back if needed
            to_squeeze = []
            for i in range(2):
                ik = len(self.shape[:-2]) + i
                if len(key) > ik and isinstance(key[ik], int):
                    to_squeeze.append(-2 + i)
            if len(to_squeeze) > 0:
                result = jnp.squeeze(result, axis=to_squeeze)

            return result


def materialize_single_matrix(
    diags: Array, offsets: (int, ...), normalized_key: (slice, ...), dtype: jnp.dtype
) -> Array:
    slice_x = normalized_key[-2]
    slice_y = normalized_key[-1]

    final_dimension = (slice_x.stop - slice_x.start, slice_y.stop - slice_y.start)

    result = jnp.zeros(final_dimension, dtype=dtype)
    if (slice_x.start == slice_x.stop) or (slice_y.start == slice_y.stop):
        return result

    for i, offset in enumerate(offsets):
        slice_in = _sparsedia_slice(offset)
        diagonal_coefficients = diags[i, slice_in]
        to_update = update_diagonal(
            diagonal_coefficients,
            slice_x.start,
            slice_x.stop,
            slice_y.start,
            slice_y.stop,
            offset,
        )

        result = result.at[
            : slice_x.stop - slice_x.start, : slice_y.stop - slice_y.start
        ].add(to_update)

    return result


def update_diagonal(L: Array, i1: int, i2: int, j1: int, j2: int, d: int = 0) -> Array:
    M = jnp.zeros((i2 - i1, j2 - j1), dtype=L.dtype)
    n = len(L)

    if (i2 <= i1) or (j2 <= j1):
        return M

    a = max(0, i1 + min(d, 0), j1 - max(d, 0))
    b = min(n, i2 + min(d, 0), j2 - max(d, 0))
    if b <= a:
        return M

    size = b - a
    row_start = a - i1 - min(d, 0)
    col_start = a - j1 + max(d, 0)

    L_diag = jnp.diag(L[a:b])
    return M.at[row_start : row_start + size, col_start : col_start + size].set(L_diag)


def normalize_slice(s: slice | int, n: int) -> slice:
    if isinstance(s, slice):
        if s.start is None:
            s = slice(0, s.stop, s.step)
        if s.stop is None:
            s = slice(s.start, n, s.step)
    else:
        s = slice(s, s + 1, None)

    return s
