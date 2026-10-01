from collections import defaultdict
from collections.abc import Sequence
from functools import partial, reduce

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.core import concrete_or_error
from jaxtyping import Array


def _sparsedia_slice(offset: int) -> slice:
    # Return the slice that selects the non-zero elements of a diagonal of given offset.
    # For example, a diagonal with offset 2 is stored as [0, 0, a, b, ..., z], and
    # _sparsedia_slice(2) will return the slice(2, None) to select [a, b, ..., z].
    return slice(offset, None) if offset >= 0 else slice(None, offset)


def transpose_sparsedia(
    offsets: tuple[int, ...], diags: Array
) -> tuple[tuple[int, ...], Array]:
    out_diags = jnp.zeros_like(diags)
    out_offsets = tuple(-x for x in offsets)

    for i, offset in enumerate(offsets):
        in_slice = _sparsedia_slice(offset)
        out_slice = _sparsedia_slice(-offset)
        out_diags = out_diags.at[..., i, out_slice].set(diags[..., i, in_slice])

    return out_offsets, out_diags


def reshape_sparsedia(
    offsets: tuple[int, ...], diags: Array, shape: tuple[int, ...]
) -> tuple[tuple[int, ...], Array]:
    shape = (*shape[:-2], len(offsets), diags.shape[-1])
    out_diags = jnp.reshape(diags, shape)
    return offsets, out_diags


def broadcast_sparsedia(
    offsets: tuple[int, ...], diags: Array, shape: tuple[int, ...]
) -> tuple[tuple[int, ...], Array]:
    shape = (*shape[:-2], len(offsets), diags.shape[-1])
    out_diags = jnp.broadcast_to(diags, shape)
    return offsets, out_diags


def powm_sparsedia(
    offsets: tuple[int, ...], diags: Array, n: int
) -> tuple[tuple[int, ...], Array]:
    if n == 0:
        out_offsets = (0,)
        out_diags = jnp.ones((*diags.shape[:-2], 1, diags.shape[-1]))
        return out_offsets, out_diags
    if n == 1:
        return offsets, diags
    else:
        n1_offsets, n1_diags = powm_sparsedia(offsets, diags, n - 1)
        return matmul_sparsedia_sparsedia(offsets, diags, n1_offsets, n1_diags)


def trace_sparsedia(offsets: tuple[int, ...], diags: Array) -> Array:
    main_diag_mask = np.asarray(offsets) == 0
    if np.any(main_diag_mask):
        return jnp.sum(diags[..., main_diag_mask, :], axis=(-1, -2))
    else:
        return jnp.zeros(diags.shape[:-2])


def mul_sparsedia_sparsedia(
    left_offsets: tuple[int, ...],
    left_diags: Array,
    right_offsets: tuple[int, ...],
    right_diags: Array,
) -> tuple[tuple[int, ...], Array]:
    # we check that the offsets are unique, as they should with the class __init__
    assert len(set(left_offsets)) == len(left_offsets)
    assert len(set(right_offsets)) == len(right_offsets)

    # compute the output offsets as the intersection of offsets
    out_offsets, left_ind, right_ind = np.intersect1d(
        left_offsets, right_offsets, assume_unique=True, return_indices=True
    )

    # initialize the output diagonals
    batch_shape = jnp.broadcast_shapes(left_diags.shape[:-2], right_diags.shape[:-2])
    out_shape = (*batch_shape, len(out_offsets), left_diags.shape[-1])
    dtype = jnp.promote_types(left_diags.dtype, right_diags.dtype)
    out_diags = jnp.zeros(out_shape, dtype=dtype)

    # loop over each output offset and fill the output
    for i in range(len(out_offsets)):
        out_diag = left_diags[..., left_ind[i], :] * right_diags[..., right_ind[i], :]
        out_diags = out_diags.at[..., i, :].set(out_diag)

    return _numpy_to_tuple(out_offsets), out_diags


def _numpy_to_tuple(x: np.ndarray) -> tuple:
    assert x.ndim == 1
    return tuple([sub_x.item() for sub_x in x])


def mul_sparsedia_array(
    offsets: tuple[int, ...], diags: Array, array: Array
) -> tuple[tuple[int, ...], Array]:
    # initialize the output diagonals
    batch_shape = jnp.broadcast_shapes(diags.shape[:-2], array.shape[:-2])
    out_shape = (*batch_shape, len(offsets), diags.shape[-1])
    dtype = jnp.promote_types(diags.dtype, array.dtype)
    out_diags = jnp.zeros(out_shape, dtype=dtype)

    # loop over each diagonal of the sparse matrix and fill the output
    for i, offset in enumerate(offsets):
        in_slice = _sparsedia_slice(offset)
        other_diag = jnp.diagonal(array, offset=offset, axis1=-2, axis2=-1)
        out_diag = other_diag * diags[..., i, in_slice]
        out_diags = out_diags.at[..., i, in_slice].set(out_diag)

    return offsets, out_diags


def tracemm_sparsedia_array(
    offsets: tuple[int, ...], diags: Array, array: Array
) -> Array:
    # tr(A @ x) = sum_o sum_j A[j - o, j] x[j, j - o]: only the diagonals of x at the
    # opposite offsets are read
    out = None
    for i, offset in enumerate(offsets):
        x_diag = jnp.diagonal(array, offset=-offset, axis1=-2, axis2=-1)
        term = (diags[..., i, _sparsedia_slice(offset)] * x_diag).sum(-1)
        out = term if out is None else out + term
    if out is None:
        batch_shape = jnp.broadcast_shapes(diags.shape[:-2], array.shape[:-2])
        dtype = jnp.promote_types(diags.dtype, array.dtype)
        return jnp.zeros(batch_shape, dtype=dtype)
    return out


def sparsedia_to_array(offsets: tuple[int, ...], diags: Array) -> Array:
    out = jnp.zeros(shape_sparsedia(diags), dtype=diags.dtype)
    for i, offset in enumerate(offsets):
        out += _vectorized_diag(diags[..., i, :], offset)
    return out


def shape_sparsedia(diags: Array) -> tuple[int, ...]:
    n = diags.shape[-1]
    return (*diags.shape[:-2], n, n)


@partial(jnp.vectorize, signature='(n)->(n,n)', excluded={1})
def _vectorized_diag(diag: Array, offset: int) -> Array:
    return jnp.diag(diag[_sparsedia_slice(offset)], k=offset)


def array_to_sparsedia(
    x: Array, offsets: tuple[int, ...] | None
) -> tuple[tuple[int, ...], Array]:
    if offsets is None:
        concrete_or_error(
            None,
            x,
            'The `offsets` argument of `array_to_sparsedia` must be statically '
            'specified to use `array_to_sparsedia` within JAX transformations.',
        )
        offsets = _find_offsets(x)

    diags = _construct_diags(offsets, x)
    return offsets, diags


def _find_offsets(x: Array) -> tuple[int, ...]:
    indices = np.nonzero(x)
    return _numpy_to_tuple(np.unique(indices[-1] - indices[-2]))


def _construct_diags(offsets: tuple[int, ...], x: Array) -> Array:
    n = x.shape[-1]
    diags = jnp.zeros((*x.shape[:-2], len(offsets), n), dtype=x.dtype)

    for i, offset in enumerate(offsets):
        diag = jnp.diagonal(x, offset=offset, axis1=-2, axis2=-1)
        diags = diags.at[..., i, _sparsedia_slice(offset)].set(diag)

    return diags


def add_sparsedia_sparsedia(
    left_offsets: tuple[int, ...],
    left_diags: Array,
    right_offsets: tuple[int, ...],
    right_diags: Array,
) -> tuple[tuple[int, ...], Array]:
    # compute the output offsets
    out_offsets = np.union1d(left_offsets, right_offsets).astype(int)

    # build each output diagonal, then stack them: XLA fuses this into one kernel,
    # while filling a zero array with `.at[i].add` is one kernel per diagonal
    batch_shape = jnp.broadcast_shapes(left_diags.shape[:-2], right_diags.shape[:-2])
    diag_shape = (*batch_shape, left_diags.shape[-1])
    dtype = jnp.promote_types(left_diags.dtype, right_diags.dtype)
    out_diags = []
    for offset in out_offsets:
        diag = jnp.zeros(diag_shape, dtype=dtype)
        if offset in left_offsets:
            diag = diag + left_diags[..., left_offsets.index(offset), :]
        if offset in right_offsets:
            diag = diag + right_diags[..., right_offsets.index(offset), :]
        out_diags.append(diag)

    if len(out_diags) == 0:
        out_diags = jnp.zeros((*batch_shape, 0, left_diags.shape[-1]), dtype=dtype)
    else:
        out_diags = jnp.stack(out_diags, axis=-2)
    return _numpy_to_tuple(out_offsets), out_diags


def matmul_sparsedia_sparsedia(
    left_offsets: tuple[int, ...],
    left_diags: Array,
    right_offsets: tuple[int, ...],
    right_diags: Array,
) -> tuple[tuple[int, ...], Array]:
    n = left_diags.shape[-1]
    batch_shape = jnp.broadcast_shapes(left_diags.shape[:-2], right_diags.shape[:-2])
    dtype = jnp.promote_types(left_diags.dtype, right_diags.dtype)
    diag_dict = defaultdict(lambda: jnp.zeros((*batch_shape, n), dtype=dtype))

    for i, loffset in enumerate(left_offsets):
        for j, roffset in enumerate(right_offsets):
            out_offset = loffset + roffset

            if abs(out_offset) > n - 1:
                continue

            lslice = _sparsedia_slice(-roffset)
            rslice = _sparsedia_slice(roffset)
            diag = left_diags[..., i, lslice] * right_diags[..., j, rslice]
            diag_dict[out_offset] = diag_dict[out_offset].at[..., rslice].add(diag)

    out_offsets = tuple(sorted(diag_dict.keys()))
    if len(out_offsets) == 0:
        # edge case where the result is a zero matrix
        out_diags = jnp.zeros((*batch_shape, 0, n), dtype=dtype)
    else:
        out_diags = jnp.stack([diag_dict[offset] for offset in out_offsets])
        out_diags = jnp.moveaxis(out_diags, 0, -2)
    return out_offsets, out_diags


# The DIA x dense products add each diagonal's contribution into the output with
# `.at[slice].add`, one scatter-add kernel per diagonal. On GPU, for operators with at
# least this many diagonals, they instead zero-pad each contribution back to full size
# and sum them, which XLA fuses into one kernel. With fewer diagonals, or on CPU, the
# scatter-adds were as fast or faster in the dynamiqs benchmarks
# (`python -m benchmarks`).
_MIN_DIAGONALS_TO_PAD = 4


def matmul_sparsedia_array(
    offsets: tuple[int, ...], diags: Array, array: Array
) -> Array:
    if len(offsets) < _MIN_DIAGONALS_TO_PAD:
        return _matmul_sparsedia_array(offsets, diags, array, pad=False)
    return lax.platform_dependent(
        diags,
        array,
        cuda=partial(_matmul_sparsedia_array, offsets, pad=True),
        default=partial(_matmul_sparsedia_array, offsets, pad=False),
    )


def _matmul_sparsedia_array(
    offsets: tuple[int, ...], diags: Array, array: Array, *, pad: bool
) -> Array:
    batch_shape = jnp.broadcast_shapes(diags.shape[:-2], array.shape[:-2])
    out_shape = (*batch_shape, diags.shape[-1], array.shape[-1])
    dtype = jnp.promote_types(diags.dtype, array.dtype)
    out = jnp.zeros(out_shape, dtype=dtype)
    for i, offset in enumerate(offsets):
        slice_in = _sparsedia_slice(offset)
        slice_out = _sparsedia_slice(-offset)
        tmp = diags[..., i, slice_in, None] * array[..., slice_in, :]
        if pad:
            out = out + _pad_rows(tmp, offset)
        else:
            out = out.at[..., slice_out, :].add(tmp)

    return out


def matmul_array_sparsedia(
    array: Array, offsets: tuple[int, ...], diags: Array
) -> Array:
    # see `_MIN_DIAGONALS_TO_PAD`
    if len(offsets) < _MIN_DIAGONALS_TO_PAD:
        return _matmul_array_sparsedia(array, offsets, diags, pad=False)
    return lax.platform_dependent(
        array,
        diags,
        cuda=lambda array, diags: _matmul_array_sparsedia(
            array, offsets, diags, pad=True
        ),
        default=lambda array, diags: _matmul_array_sparsedia(
            array, offsets, diags, pad=False
        ),
    )


def _matmul_array_sparsedia(
    array: Array, offsets: tuple[int, ...], diags: Array, *, pad: bool
) -> Array:
    batch_shape = jnp.broadcast_shapes(array.shape[:-2], diags.shape[:-2])
    out_shape = (*batch_shape, array.shape[-2], diags.shape[-1])
    dtype = jnp.promote_types(array.dtype, diags.dtype)
    out = jnp.zeros(out_shape, dtype=dtype)
    for i, offset in enumerate(offsets):
        slice_in = _sparsedia_slice(offset)
        slice_out = _sparsedia_slice(-offset)
        tmp = array[..., :, slice_out] * diags[..., i, None, slice_in]
        if pad:
            out = out + _pad_columns(tmp, offset)
        else:
            out = out.at[..., :, slice_in].add(tmp)

    return out


def _pad_rows(x: Array, offset: int) -> Array:
    # Zero-pad the second-to-last axis of `x` back to full size: the rows selected by
    # `_sparsedia_slice(-offset)`.
    pad_width = [(0, 0)] * x.ndim
    pad_width[-2] = (0, offset) if offset >= 0 else (-offset, 0)
    return jnp.pad(x, pad_width)


def _pad_columns(x: Array, offset: int) -> Array:
    # Zero-pad the last axis of `x` back to full size: the columns selected by
    # `_sparsedia_slice(offset)`.
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (offset, 0) if offset >= 0 else (0, -offset)
    return jnp.pad(x, pad_width)


def and_sparsedia_sparsedia(
    left_offsets: tuple[int, ...],
    left_diags: Array,
    right_offsets: tuple[int, ...],
    right_diags: Array,
) -> tuple[tuple[int, ...], Array]:
    # compute new offsets
    n = right_diags.shape[-1]
    left_offsets_np = np.asarray(left_offsets)
    right_offsets_np = np.asarray(right_offsets)
    out_offsets = _numpy_to_tuple(
        np.ravel(left_offsets_np[:, None] * n + right_offsets_np)
    )

    # compute new diagonals with broadcasted batch axes
    out_diags = _bkron(left_diags, right_diags)

    # merge duplicate offsets and return
    out_offsets, out_diags = _compress_sparsedia(out_offsets, out_diags)
    return out_offsets, out_diags


@partial(jnp.vectorize, signature='(a,b),(c,d)->(ac,bd)')
def _bkron(a: Array, b: Array) -> Array:
    return jnp.kron(a, b)


def _compress_sparsedia(
    offsets: tuple[int, ...], diags: Array
) -> tuple[tuple[int, ...], Array]:
    # compute unique offsets
    out_offsets, inverse_ind = np.unique(offsets, return_inverse=True)

    # initialize output diagonals
    diags_shape = (*diags.shape[:-2], len(out_offsets), diags.shape[-1])
    out_diags = jnp.zeros(diags_shape, dtype=diags.dtype)

    # loop over each offset and fill the output
    for i in range(len(out_offsets)):
        mask = inverse_ind == i
        diag = jnp.sum(diags[..., mask, :], axis=-2)
        out_diags = out_diags.at[..., i, :].set(diag)

    return _numpy_to_tuple(out_offsets), out_diags


def stack_sparsedia(
    offsets_sequence: Sequence[tuple[int, ...]],
    diags_sequence: Sequence[Array],
    axis: int,
) -> tuple[tuple[int, ...], Array]:
    # compute unique offsets of the output
    out_offsets = np.unique(np.concatenate(offsets_sequence))
    offset_to_index = {offset: idx for idx, offset in enumerate(out_offsets)}

    # prepare output diagonals with the correct shape and dtype
    dtype = reduce(jnp.promote_types, [diags.dtype for diags in diags_sequence])
    in_shape = diags_sequence[0].shape
    out_shape = (len(diags_sequence), *in_shape[:-2], len(out_offsets), in_shape[-1])
    out_diags = jnp.zeros(out_shape, dtype=dtype)
    for i, (offsets, diags) in enumerate(
        zip(offsets_sequence, diags_sequence, strict=True)
    ):
        for j, offset in enumerate(offsets):
            idx = offset_to_index[offset]
            out_diags = out_diags.at[i, ..., idx, :].add(diags[..., j, :])

    # move the stack axis to the correct position
    out_diags = jnp.moveaxis(out_diags, 0, axis)
    return _numpy_to_tuple(out_offsets), out_diags


def concatenate_sparsedia(
    offsets_sequence: Sequence[tuple[int, ...]],
    diags_sequence: Sequence[Array],
    axis: int,
) -> tuple[tuple[int, ...], Array]:
    # compute unique offsets of the output
    out_offsets = np.asarray(sorted(reduce(set.union, map(set, offsets_sequence))))
    offset_to_index = {offset: idx for idx, offset in enumerate(out_offsets)}

    # prepare input diagonals with matching offsets before concatenating
    dtype = reduce(jnp.promote_types, [diags.dtype for diags in diags_sequence])
    expanded_diags = []
    for offsets, diags in zip(offsets_sequence, diags_sequence, strict=True):
        out_shape = (*diags.shape[:-2], len(out_offsets), diags.shape[-1])
        out_diags = jnp.zeros(out_shape, dtype=dtype)
        for i, offset in enumerate(offsets):
            idx = offset_to_index[offset]
            out_diags = out_diags.at[..., idx, :].add(diags[..., i, :])
        expanded_diags.append(out_diags)

    diags = jnp.concatenate(expanded_diags, axis=axis)
    return _numpy_to_tuple(out_offsets), diags


def autopad_sparsedia_diags(offsets: tuple[int, ...], diags: Sequence[Array]) -> Array:
    # stack diags in a square matrix by padding each according to its offset
    pads_width = [(abs(k), 0) if k >= 0 else (0, abs(k)) for k in offsets]
    diags = [
        jnp.pad(diag, pad_width)
        for pad_width, diag in zip(pads_width, diags, strict=True)
    ]
    dtype = reduce(jnp.promote_types, [diag.dtype for diag in diags])
    stacked_diags = jnp.stack(diags, dtype=dtype)
    return jnp.moveaxis(stacked_diags, 0, -2)


# Most shifts (p, q) for which `lindbladian_sparsedia` is used (see
# `lindbladian_sparsedia_shifts`). Each shift reads rho once more: on an A100, with 16
# shifts (n = 1176, batch 16) the fused sum is 5.6x faster than the products, with 71
# (n = 900, batch 4: rho no longer fits in L2) the products are 1.3x faster.
MAX_FUSED_SHIFTS = 32


def lindbladian_sparsedia_shifts(
    left_offsets: tuple[int, ...],
    right_offsets: tuple[int, ...],
    jump_offsets: Sequence[tuple[int, ...]],
) -> int:
    # number of distinct shifted copies of rho that `lindbladian_sparsedia` reads
    shifts = {(o, 0) for o in left_offsets} | {(0, -o) for o in right_offsets}
    for offsets in jump_offsets:
        shifts |= {(a, b) for a in offsets for b in offsets}
    return len(shifts)


def lindbladian_sparsedia(
    left: tuple[tuple[int, ...], Array],
    right: tuple[tuple[int, ...], Array],
    jump_ops: Sequence[tuple[tuple[int, ...], Array]],
    rho: Array,
) -> Array:
    r"""Return $A\rho + \rho B + \sum_k L_k \rho L_k^\dag$ for a dense $\rho$ and
    operators $A$, $B$ and $L_k$ in DIA format, each given as (offsets, diags).

    Every term is a shifted copy of $\rho$ weighted by row and column coefficients,
    $R[i]\,\rho[i+p, j+q]\,C[j]$, with $(p, q) = (o, 0)$ for each diagonal $o$ of $A$,
    $(0, -o)$ for each diagonal of $B$, and $(a, b)$ for each pair of diagonals of an
    $L_k$. The terms are grouped by shift and summed in one elementwise expression,
    which XLA fuses into a single pass over $\rho$, instead of a pass per product.
    """
    n = rho.shape[-1]
    operators = [left, right, *jump_ops]
    pad = max((abs(o) for offsets, _ in operators for o in offsets), default=0)
    terms = _lindbladian_sparsedia_terms(left, right, jump_ops, n, pad)
    widths = [(0, 0)] * (rho.ndim - 2) + [(pad, pad), (pad, pad)]
    rho_padded = jnp.pad(rho, widths)

    def term(p: int, q: int) -> Array:
        block = rho_padded[..., pad + p : pad + p + n, pad + q : pad + q + n]
        weight = 0
        for row, column in terms[p, q]:
            if row is not None and column is not None:
                weight = weight + row[..., :, None] * column[..., None, :]
            elif row is not None:
                weight = weight + row[..., :, None]
            elif column is not None:
                weight = weight + column[..., None, :]
        return weight * block

    # Each shift (p, q) is summed with its mirror (q, p). When rho is Hermitian and
    # B = A^dag, the terms at (j, i) are then the exact complex conjugates of those at
    # (i, j), added in the same order: the result is exactly Hermitian, as the
    # Hermitian form tmp + tmp^dag is by construction.
    out = jnp.zeros_like(rho)
    done = set()
    for p, q in terms:
        if (p, q) in done:
            continue
        done |= {(p, q), (q, p)}
        if p == q or (q, p) not in terms:
            out = out + term(p, q)
        else:
            out = out + (term(p, q) + term(q, p))
    return out


def _lindbladian_sparsedia_terms(
    left: tuple[tuple[int, ...], Array],
    right: tuple[tuple[int, ...], Array],
    jump_ops: Sequence[tuple[tuple[int, ...], Array]],
    n: int,
    pad: int,
) -> dict[tuple[int, int], list[tuple[Array | None, Array | None]]]:
    # the terms of `lindbladian_sparsedia`, grouped by shift (p, q): a list of (row
    # coefficients or None, column coefficients or None)
    def shifted(diag: Array, offset: int) -> Array:
        # v[i] = diag[i + offset], 0 outside [0, n)
        widths = [(0, 0)] * (diag.ndim - 1) + [(pad, pad)]
        return jnp.pad(diag, widths)[..., pad + offset : pad + offset + n]

    terms = defaultdict(list)
    offsets, diags = left
    for k, offset in enumerate(offsets):  # A[i, i+o] rho[i+o, j]
        terms[offset, 0].append((shifted(diags[..., k, :], offset), None))
    offsets, diags = right
    for k, offset in enumerate(offsets):  # rho[i, j-o] B[j-o, j]
        terms[0, -offset].append((None, diags[..., k, :]))
    for offsets, diags in jump_ops:  # L[i, i+a] rho[i+a, j+b] conj(L[j, j+b])
        rows = [shifted(diags[..., k, :], o) for k, o in enumerate(offsets)]
        for a, row in zip(offsets, rows, strict=True):
            for b, column in zip(offsets, rows, strict=True):
                terms[a, b].append((row, column.conj()))
    return terms
