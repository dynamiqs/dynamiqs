from __future__ import annotations

import warnings
from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, DTypeLike
from qutip import Qobj

from .._checks import check_shape
from .dense_qarray import DenseQArray, _dense_to_qobj
from .layout import Layout, dense, get_layout
from .qarray import QArray, QArrayLike, _dims_from_qutip, _dims_to_qutip, _to_jax
from .sparsedia_qarray import (
    SparseDIAQArray,
    _array_to_sparsedia,
    _sparsedia_to_dense,
    _sparsedia_to_qobj,
)
from .utils import stack

__all__ = ['asqarray', 'asdense', 'assparsedia', 'asjaxarray', 'asqobj', 'sparsedia']


def asqarray(
    x: QArrayLike, dims: tuple[int, ...] | None = None, layout: Layout = None
) -> QArray:
    layout = get_layout(layout)
    if layout is dense:
        return _asdense(x, dims=dims)
    else:
        return _assparsedia(x, dims=dims)


def _asdense(x: QArrayLike, dims: tuple[int, ...] | None = None) -> DenseQArray:
    _warn_qarray_dims(x, dims)

    if isinstance(x, DenseQArray):
        return x
    elif isinstance(x, SparseDIAQArray):
        return _sparsedia_to_dense(x)
    elif isinstance(x, Qobj):
        dims = _dims_from_qutip(dims)
        x = x.full()
    elif isinstance(x, Sequence) and all(isinstance(sub_x, QArray) for sub_x in x):
        # TODO: generalize to any nested sequence with the appropriate shape
        return stack([_asdense(sub_x, dims=dims) for sub_x in x])

    x = jnp.asarray(x)
    dims = _init_dims(x, dims)
    return DenseQArray(dims, x)


def _assparsedia(x: QArrayLike, dims: tuple[int, ...] | None = None) -> SparseDIAQArray:
    _warn_qarray_dims(x, dims)

    if isinstance(x, SparseDIAQArray):
        return x
    elif isinstance(x, DenseQArray):
        dims = x.dims
        x = x.asjaxarray()
    elif isinstance(x, Qobj):
        # TODO: improve this by directly extracting the diags and offsets in case
        # the Qobj is already in sparse DIA format (only for QuTiP 5)
        dims = _dims_from_qutip(dims)
        x = x.full()
    elif isinstance(x, Sequence) and all(isinstance(sub_x, QArray) for sub_x in x):
        # TODO: generalize to any nested sequence with the appropriate shape
        return stack([_assparsedia(sub_x, dims=dims) for sub_x in x])

    x = jnp.asarray(x)
    dims = _init_dims(x, dims)
    return _array_to_sparsedia(x, dims=dims)


def asjaxarray(x: QArrayLike) -> Array:
    return _asjaxarray(x)


def asqobj(x: QArrayLike, dims: tuple[int, ...] | None = None) -> Qobj | list[Qobj]:
    r"""Convert a qarray-like object into a QuTiP Qobj (or a list of QuTiP Qobj if it
    has more than two dimensions).

    Args:
        x _(qarray_like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra,
            density matrix or operator.
        dims _(tuple of ints or None)_: Dimensions of each subsystem in the large
            Hilbert space of the composite system, defaults to `None` (a single system
            with the same dimension as `x`).

    Returns:
        QuTiP Qobj or list of QuTiP Qobj.

    Examples:
        >>> psi = dq.fock(3, 1)
        >>> psi
        QArray: shape=(3, 1), dims=(3,), dtype=complex64, layout=dense
        [[0.+0.j]
         [1.+0.j]
         [0.+0.j]]
        >>> dq.asqobj(psi)
        Quantum object: dims=[[3], [1]], shape=(3, 1), type='ket', dtype=Dense
        Qobj data =
        [[0.]
         [1.]
         [0.]]

        For a batched array:
        >>> rhos = dq.stack([dq.coherent_dm(16, i) for i in range(5)])
        >>> rhos.shape
        (5, 16, 16)

        # todo: temporary fix
        # >>> len(dq.asqobj(rhos))
        # 5

        Note that the tensor product structure is inferred automatically for qarrays. It
        can be specified with the `dims` argument for other types.
        >>> I = dq.eye(3, 2)
        >>> dq.asqobj(I)
        Quantum object: dims=[[3, 2], [3, 2]], shape=(6, 6), type='oper', dtype=Dense, isherm=True
        Qobj data =
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]
    """  # noqa: E501
    _warn_qarray_dims(x, dims)

    if isinstance(x, Qobj):
        return x
    elif isinstance(x, DenseQArray):
        return _dense_to_qobj(x)
    elif isinstance(x, SparseDIAQArray):
        return _sparsedia_to_qobj(x)
    elif isinstance(x, Sequence) and all(isinstance(sub_x, QArray) for sub_x in x):
        # TODO: generalize to any nested sequence with the appropriate shape
        return [asqobj(sub_x, dims=dims) for sub_x in x]

    x = jnp.asarray(x)
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')
    dims = _init_dims(x, dims)
    dims = _dims_to_qutip(dims, x.shape)
    return Qobj(x, dims=dims)


def _warn_qarray_dims(x: QArrayLike, dims: tuple[int, ...] | None = None):
    if dims is not None:
        if isinstance(x, QArray) and x.dims != dims:
            warnings.warn(
                f'Argument `x` is already a QArray with dims={x.dims}, but dims '
                f'were also provided as input with dims={dims}. Ignoring the '
                'provided `dims` and proceeding with `x.dims`.',
                stacklevel=2,
            )
        elif isinstance(x, Qobj) and _dims_from_qutip(x.dims) != dims:
            warnings.warn(
                f'Argument `x` is already a Qobj with dims={x.dims}, but dims '
                f'were also provided as input with dims={dims}. Ignoring the '
                'provided `dims` and proceeding with `x.dims`.',
                stacklevel=2,
            )


def sparsedia(
    offsets_diags: dict[int, ArrayLike],
    dims: tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
) -> SparseDIAQArray:
    # === offsets
    offsets = tuple(offsets_diags.keys())

    # === diags
    # stack arrays in a square matrix by padding each according to its offset
    pads_width = [(abs(k), 0) if k >= 0 else (0, abs(k)) for k in offsets]
    diags = [jnp.asarray(diag) for diag in offsets_diags.values()]
    diags = [jnp.pad(diag, pad_width) for pad_width, diag in zip(pads_width, diags)]
    diags = jnp.stack(diags, dtype=dtype)

    # === dims
    n = diags.shape[-1]
    shape = (*diags.shape[:-2], n, n)
    dims = (n,) if dims is None else dims
    _check_dims_match_shape(shape, dims)

    return SparseDIAQArray(diags=diags, offsets=offsets, dims=dims)


def _init_dims(x: Array, dims: tuple[int, ...] | None = None) -> tuple[int, ...]:
    if dims is None:
        dims = (x.shape[-2],) if x.shape[-2] != 1 else (x.shape[-1],)

    _check_dims_match_shape(x.shape, dims)

    # TODO: check if is bra, ket, dm or op
    # if not (isbra(data) or isket(data) or isdm(data) or isop(data)):
    # raise ValueError(
    #     f'DenseQArray data must be a bra, a ket, a density matrix '
    #     f'or and operator. Got array with size {data.shape}'
    # )

    return dims


def _check_dims_match_shape(shape: tuple[int, ...], dims: tuple[int, ...]):
    if np.prod(dims) != np.max(shape[-2:]):
        raise ValueError(
            'The provided `dims` are incompatible with the input array. '
            f'Got dims={dims} and shape={shape}.'
        )
