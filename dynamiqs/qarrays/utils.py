from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import cast

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, DTypeLike
from qutip import Qobj

from .dense_dataarray import DenseDataArray, array_to_qobj_list
from .layout import Layout, dense
from .materialized_qarray import MaterializedQArray
from .qarray import (
    QArray,
    QArrayLike,
    check_compatible_dims,
    get_dims,
    isqarraylike,
    to_jax,
    to_numpy,
)
from .sparsedia_dataarray import SparseDIADataArray
from .sparsedia_primitives import (
    array_to_sparsedia,
    autopad_sparsedia_diags,
    concatenate_sparsedia,
    shape_sparsedia,
    stack_sparsedia,
)

__all__ = [
    'asqarray',
    'concatenate',
    'expand_dims',
    'moveaxis',
    'sparsedia_from_dict',
    'stack',
    'swapaxes',
    'where',
    'to_jax',
    'to_numpy',
    'to_qutip',
    'isqarraylike',
]


def asqarray(
    x: QArrayLike,
    dims: tuple[int, ...] | None = None,
    layout: Layout | None = None,
    *,
    offsets: tuple[int, ...] | None = None,
) -> QArray:
    """Converts a qarray-like into a qarray.

    Args:
        x: Object to convert.
        dims (tuple of ints or None): Hilbert space dimension of each subsystem.
            Defaults to `None` (`x.dims` if available, individual system `dims=(n,)`
            otherwise).
        layout (dq.dense, dq.dia or None): Matrix layout. If `None`, the default
            layout is `dq.dense`, except for qarrays that are directly returned.
        offsets: Offsets of the stored diagonals if `layout==dq.dia`. If `None`, offsets
            are determined automatically from the matrix structure. This argument can
            also be explicitly specified to ensure compatibility with JAX
            transformations, which require static offset values.

    Returns:
        Qarray representation of the input.

    See also:
        - [`dq.isqarraylike()`][dynamiqs.isqarraylike]: returns True if the input is
            a qarray-like.

    Examples:
        >>> dq.asqarray([[1, 0], [0, -1]])
        QArray: shape=(2, 2), dims=(2,), dtype=int32, layout=dense
        [[ 1  0]
         [ 0 -1]]
        >>> dq.asqarray([[1, 0], [0, -1]], layout=dq.dia)
        QArray: shape=(2, 2), dims=(2,), dtype=int32, layout=dia, ndiags=1
        [[ 1  ⋅]
         [ ⋅ -1]]
        >>> dq.asqarray([qt.sigmax(), qt.sigmay(), qt.sigmaz()])
        QArray: shape=(3, 2, 2), dims=(2,), dtype=complex64, layout=dense
        [[[ 0.+0.j  1.+0.j]
          [ 1.+0.j  0.+0.j]]
        <BLANKLINE>
         [[ 0.+0.j  0.-1.j]
          [ 0.+1.j  0.+0.j]]
        <BLANKLINE>
         [[ 1.+0.j  0.+0.j]
          [ 0.+0.j -1.+0.j]]]
    """
    if layout is None and isinstance(x, QArray):
        return x

    layout = dense if layout is None else layout
    if layout is dense:
        return _asdense(x, dims)
    else:
        return _assparsedia(x, dims, offsets)


def _asdense(x: QArrayLike, dims: tuple[int, ...] | None) -> QArray:
    if isinstance(x, MaterializedQArray):
        if isinstance(x.data, DenseDataArray):
            return x
        else:
            return x.asdense()

    xdims = get_dims(x)
    x = to_jax(x)
    dims = init_dims(xdims, dims, x.shape)
    return MaterializedQArray(dims, False, DenseDataArray(x))


def _assparsedia(
    x: QArrayLike, dims: tuple[int, ...] | None, offsets: tuple[int, ...] | None
) -> QArray:
    # TODO: improve this by directly extracting the diags and offsets in case
    # the Qobj is already in sparse DIA format (only for QuTiP 5)
    if isinstance(x, MaterializedQArray):
        if isinstance(x.data, SparseDIADataArray):
            return x
        # convert dense to sparse
        xdims = x.dims
        x_jax = x.to_jax()
        dims = init_dims(xdims, dims, x_jax.shape)
        offsets, diags = array_to_sparsedia(x_jax, offsets)
        return MaterializedQArray(dims, False, SparseDIADataArray(offsets, diags))

    xdims = get_dims(x)
    x = to_jax(x)
    dims = init_dims(xdims, dims, x.shape)
    offsets, diags = array_to_sparsedia(x, offsets)
    return MaterializedQArray(dims, False, SparseDIADataArray(offsets, diags))


def init_dims(
    xdims: tuple[int, ...] | None, dims: tuple[int, ...] | None, shape: tuple[int, ...]
) -> tuple[int, ...]:
    # xdims: native dims from the original object
    # dims: dims specified by the user
    # shape: object shape
    if dims is None:
        dims = (shape[-2] if shape[-2] != 1 else shape[-1],) if xdims is None else xdims
    elif xdims is not None and xdims != dims:
        # warn if `dims` argument is specified but unused
        warnings.warn(
            f'Argument `x` is already an object with `x.dims={xdims}`, but'
            f' different `dims={dims}` were specified as input. Ignoring the '
            f'provided `dims` and proceeding with the object `x.dims`.',
            stacklevel=2,
        )

    _assert_dims_match_shape(dims, shape)

    return dims


def stack(qarrays: Sequence[QArray], axis: int = 0) -> QArray:
    """Join a sequence of qarrays along a new axis.

    Warning:
        All elements of the sequence `qarrays` must have identical types, shapes and
        `dims` attributes. Additionally, when stacking qarrays with sparse diagonal
        data, all elements must have identical `offsets` attributes.

    Args:
        qarrays: Qarrays to stack.
        axis: Axis in the result along which the input qarrays are stacked.

    Returns:
        Stacked qarray.

    Examples:
        >>> dq.stack([dq.fock(3, 0), dq.fock(3, 1)])
        QArray: shape=(2, 3, 1), dims=(3,), dtype=complex64, layout=dense
        [[[1.+0.j]
          [0.+0.j]
          [0.+0.j]]
        <BLANKLINE>
         [[0.+0.j]
          [1.+0.j]
          [0.+0.j]]]
    """
    # check validity of input
    if len(qarrays) == 0:
        raise ValueError('Argument `qarrays` must contain at least one element.')
    if not all(isinstance(q, QArray) for q in qarrays):
        raise ValueError(
            'Argument `qarrays` must contain only elements of type `QArray`.'
        )
    dims = qarrays[0].dims
    if not all(q.dims == dims for q in qarrays):
        raise ValueError(
            'Argument `qarrays` must contain elements with identical `dims` attribute.'
        )
    if not all(qarray.shape == qarrays[0].shape for qarray in qarrays):
        raise ValueError('All input qarrays must have the same shape.')

    # stack inputs depending on data type
    if all(
        isinstance(q, MaterializedQArray) and isinstance(q.data, DenseDataArray)
        for q in qarrays
    ):
        _qarrays = cast(list[MaterializedQArray], qarrays)
        data = jnp.stack(
            [cast(DenseDataArray, q.data).data for q in _qarrays], axis=axis
        )
        return MaterializedQArray(dims, False, DenseDataArray(data))

    elif all(
        isinstance(q, MaterializedQArray) and isinstance(q.data, SparseDIADataArray)
        for q in qarrays
    ):
        _qarrays = cast(list[MaterializedQArray], qarrays)
        offsets, diags = stack_sparsedia(
            [cast(SparseDIADataArray, q.data).offsets for q in _qarrays],
            [cast(SparseDIADataArray, q.data).diags for q in _qarrays],
            axis=axis,
        )
        return MaterializedQArray(dims, False, SparseDIADataArray(offsets, diags))
    else:
        raise NotImplementedError(
            'Stacking qarrays with different data types is not implemented.'
        )


def swapaxes(x: QArrayLike, axis1: int, axis2: int) -> QArray:
    """Interchange two axes of a qarray.

    Args:
        x: Qarray-like object.
        axis1: First axis.
        axis2: Second axis.

    Returns:
        Qarray with axes `axis1` and `axis2` interchanged.
    """
    return asqarray(x).swapaxes(axis1, axis2)


def moveaxis(
    x: QArrayLike, source: int | Sequence[int], destination: int | Sequence[int]
) -> QArray:
    """Move axes of a qarray to new positions.

    Args:
        x: Qarray-like object.
        source: Original positions of the axes to move.
        destination: Destination positions for each moved axis.

    Returns:
        Qarray with moved axes.
    """
    return asqarray(x).moveaxis(source, destination)


def expand_dims(x: QArrayLike, axis: int | Sequence[int]) -> QArray:
    """Expand the shape of a qarray by inserting new axes.

    Args:
        x: Qarray-like object.
        axis: Axis or axes where new dimensions are inserted.

    Returns:
        Qarray with additional dimensions.
    """
    return asqarray(x).expand_dims(axis)


def where(condition: ArrayLike, x: QArrayLike, y: QArrayLike) -> QArray:
    """Select values from two qarrays depending on a condition.

    Args:
        condition: Boolean array-like condition.
        x: Values selected when `condition` is true.
        y: Values selected when `condition` is false.

    Returns:
        Qarray with values chosen from `x` and `y`.
    """
    x_is_qarray = isinstance(x, QArray)
    y_is_qarray = isinstance(y, QArray)

    if not x_is_qarray and not y_is_qarray:
        raise TypeError(
            'At least one of `x` or `y` must be a QArray. For non-qarray operands, '
            'use `jax.numpy.where` directly.'
        )

    if isinstance(x, MaterializedQArray) and isinstance(y, MaterializedQArray):
        _check_compatible_qarray_metadata(x, y)
        dims = x.dims
        vectorized = x.vectorized
    elif isinstance(x, MaterializedQArray):
        y_dims = get_dims(y)
        if y_dims is not None:
            check_compatible_dims(x.dims, y_dims)
        dims = x.dims
        vectorized = x.vectorized
    elif isinstance(y, MaterializedQArray):
        x_dims = get_dims(x)
        if x_dims is not None:
            check_compatible_dims(y.dims, x_dims)
        dims = y.dims
        vectorized = y.vectorized
    else:
        raise NotImplementedError(
            '`where` is only implemented for materialized qarrays.'
        )

    if _contains_sparse_qarray(x, y):
        # TODO: Generalize implementation of `dq.where` to sparse dia layout.
        _warn_sparse_to_dense('where')

    data = jnp.where(condition, to_jax(x), to_jax(y))
    return MaterializedQArray(dims, vectorized, DenseDataArray(data))


def concatenate(qarrays: Sequence[QArray], axis: int = 0) -> QArray:
    """Join a sequence of qarrays along an existing axis.

    Args:
        qarrays: Qarrays to concatenate.
        axis: Axis in the result along which the input qarrays are concatenated.

    Returns:
        Concatenated qarray.
    """
    qarrays, dims, vectorized, axis = _check_concatenate_input(qarrays, axis)

    if all(isinstance(q.data, DenseDataArray) for q in qarrays):
        data = DenseDataArray(
            jnp.concatenate([cast(DenseDataArray, q.data).data for q in qarrays], axis)
        )
    elif all(isinstance(q.data, SparseDIADataArray) for q in qarrays):
        offsets, diags = concatenate_sparsedia(
            [cast(SparseDIADataArray, q.data).offsets for q in qarrays],
            [cast(SparseDIADataArray, q.data).diags for q in qarrays],
            axis=axis,
        )
        data = SparseDIADataArray(offsets, diags)
    else:
        _warn_sparse_to_dense('concatenate')
        data = DenseDataArray(jnp.concatenate([q.to_jax() for q in qarrays], axis))

    return MaterializedQArray(dims, vectorized, data)


def _check_concatenate_input(
    qarrays: Sequence[QArray], axis: int
) -> tuple[list[MaterializedQArray], tuple[int, ...], bool, int]:
    if len(qarrays) == 0:
        raise ValueError('Argument `qarrays` must contain at least one element.')
    if not all(isinstance(q, MaterializedQArray) for q in qarrays):
        raise ValueError(
            'Argument `qarrays` must contain only elements of type `QArray`.'
        )

    materialized_qarrays = cast(list[MaterializedQArray], qarrays)
    dims = materialized_qarrays[0].dims
    vectorized = materialized_qarrays[0].vectorized
    ndim = materialized_qarrays[0].ndim
    if not -ndim <= axis < ndim:
        raise ValueError(f'axis {axis} is out of bounds for array of dimension {ndim}')
    axis = axis % ndim
    if axis >= ndim - 2:
        raise ValueError(
            '`concatenate` can only manipulate batching dimensions of a qarray; the '
            'final two dimensions are matrix dimensions.'
        )

    if not all(q.dims == dims for q in materialized_qarrays):
        raise ValueError(
            'Argument `qarrays` must contain elements with identical `dims` attribute.'
        )
    if not all(q.ndim == ndim for q in materialized_qarrays):
        raise ValueError(
            'Argument `qarrays` must contain elements with the same number of '
            'batching dimensions.'
        )
    if not all(q.shape[-2:] == materialized_qarrays[0].shape[-2:] for q in qarrays):
        raise ValueError(
            'Argument `qarrays` must contain elements with identical final two '
            'dimensions.'
        )
    if not all(q.vectorized == vectorized for q in materialized_qarrays):
        raise ValueError(
            'Argument `qarrays` must contain elements with identical `vectorized` '
            'attribute.'
        )

    return materialized_qarrays, dims, vectorized, axis


def to_qutip(x: QArrayLike, dims: tuple[int, ...] | None = None) -> Qobj | list[Qobj]:
    r"""Convert a qarray-like into a QuTiP Qobj or list of Qobjs.

    Args:
        x (qarray-like of shape (..., n, 1) or (..., 1, n) or (..., n, n)): Ket, bra,
            density matrix or operator.
        dims (tuple of ints or None): Hilbert space dimension of each subsystem.
            Defaults to `None` (`x.dims` if available, individual system `dims=(n,)`
            otherwise).

    Returns:
        QuTiP Qobj or list of QuTiP Qobj.

    Examples:
        >>> dq.fock(3, 1)
        QArray: shape=(3, 1), dims=(3,), dtype=complex64, layout=dense
        [[0.+0.j]
         [1.+0.j]
         [0.+0.j]]
        >>> dq.to_qutip(dq.fock(3, 1))
        Quantum object: dims=[[3], [1]], shape=(3, 1), type='ket', dtype=Dense
        Qobj data =
        [[0.]
         [1.]
         [0.]]

        For a batched qarray:
        >>> rhos = dq.stack([dq.coherent_dm(16, i) for i in range(5)])
        >>> rhos.shape
        (5, 16, 16)
        >>> len(dq.to_qutip(rhos))
        5

        Note that the tensor product structure is inferred automatically for qarrays. It
        can be specified with the `dims` argument for other types.
        >>> dq.to_qutip(dq.eye(3, 2))
        Quantum object: dims=[[3, 2], [3, 2]], shape=(6, 6), type='oper', dtype=Dense, isherm=True
        Qobj data =
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]
    """  # noqa: E501
    from .._checks import check_shape  # noqa: PLC0415

    if isinstance(x, Qobj):
        return x
    elif isinstance(x, QArray):
        return x.to_qutip()

    xdims = get_dims(x)
    x = to_jax(x)
    dims = init_dims(xdims, dims, x.shape)
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')
    return array_to_qobj_list(x, dims)


def sparsedia_from_dict(
    offsets_diags: dict[int, ArrayLike],
    dims: tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
) -> QArray:
    """Initialize a sparse diagonal qarray from a dictionary of offsets and diagonals.

    Args:
        offsets_diags: Dictionary where keys are offsets and values are diagonals of
            shapes _(..., n-|offset|)_ with a common batch shape between all diagonals.
        dims (tuple of ints or None): Hilbert space dimension of each subsystem.
            Defaults to `None` (`x.dims` if available, individual system `dims=(n,)`
            otherwise).
        dtype: Data type of the qarray. If `None`, the data type is inferred from the
            diagonals.

    Returns:
        A sparse diagonal qarray with non-zero diagonals at the specified offsets.

    Examples:
        >>> dq.sparsedia_from_dict({0: [1, 2, 3], 1: [4, 5], -1: [6, 7]})
        QArray: shape=(3, 3), dims=(3,), dtype=int32, layout=dia, ndiags=3
        [[1 4 ⋅]
         [6 2 5]
         [⋅ 7 3]]
        >>> dq.sparsedia_from_dict({0: jnp.ones((3, 2))})
        QArray: shape=(3, 2, 2), dims=(2,), dtype=float32, layout=dia, ndiags=1
        [[[1. ⋅ ]
          [ ⋅ 1.]]
        <BLANKLINE>
         [[1. ⋅ ]
          [ ⋅ 1.]]
        <BLANKLINE>
         [[1. ⋅ ]
          [ ⋅ 1.]]]
    """
    offsets = tuple(offsets_diags.keys())
    diags = [jnp.asarray(diag, dtype=dtype) for diag in offsets_diags.values()]
    diags = autopad_sparsedia_diags(offsets, diags)
    shape = shape_sparsedia(diags)
    dims = (shape[-1],) if dims is None else dims
    _assert_dims_match_shape(dims, shape)

    return MaterializedQArray(dims, False, SparseDIADataArray(offsets, diags))


def _assert_dims_match_shape(dims: tuple[int, ...], shape: tuple[int, ...]):
    # check that `dims` and `shape` are compatible
    if np.prod(dims) != np.max(shape[-2:]):
        raise ValueError(
            f'Argument `dims={dims}` is incompatible with the input shape'
            f' `shape={shape}`.'
        )


def _contains_sparse_qarray(*xs: QArrayLike) -> bool:
    return any(
        isinstance(x, MaterializedQArray) and isinstance(x.data, SparseDIADataArray)
        for x in xs
    )


def _check_compatible_qarray_metadata(
    x: MaterializedQArray, y: MaterializedQArray
) -> None:
    check_compatible_dims(x.dims, y.dims)
    if x.shape[-2:] != y.shape[-2:]:
        raise ValueError(
            'Qarrays have incompatible final two dimensions. '
            f'Got {x.shape[-2:]} and {y.shape[-2:]}.'
        )
    if x.vectorized != y.vectorized:
        raise ValueError(
            'Qarrays have incompatible `vectorized` attributes. '
            f'Got {x.vectorized} and {y.vectorized}.'
        )


def _warn_sparse_to_dense(operation: str) -> None:
    warnings.warn(
        'A sparse qarray has been converted to dense layout while applying '
        f'`{operation}`.',
        stacklevel=2,
    )
