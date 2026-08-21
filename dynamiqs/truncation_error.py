r"""A posteriori Fock truncation error estimate for the Lindblad master equation.

Implements the space-truncation estimator of
[arXiv:2501.09607](https://arxiv.org/abs/2501.09607) (*A posteriori error estimates for
the Lindblad master equation*, Etienney, Robin, Rouchon):
$$
    \xi(t) = \int_0^t \| (\mathcal{L} - \mathcal{L}_N)\rho_N(s) \|_1 \dd s
           \geq \| \rho(t) - \rho_N(t) \|_1,
$$
where $\rho_N$ is the simulated state on the truncated space and
$\mathcal{L}$/$\mathcal{L}_N$ are the exact and truncated Lindbladians.

For a Hamiltonian and jump operators that are polynomials in the ladder operators, the
residual $(\mathcal{L} - \mathcal{L}_N)\rho_N$ is exactly supported on the Fock space
extended by a finite per-mode buffer. Both that buffer and the operators on the extended
space are derived here from the truncated operators themselves, see
`extension_buffer()` and `extend_timeqarray()`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from diffrax._custom_types import RealScalarLike
from jax import Array

from .qarrays.layout import dia
from .qarrays.qarray import QArray
from .qarrays.sparsedia_dataarray import SparseDIADataArray
from .qarrays.utils import sparsedia_from_dict
from .time_qarray import (
    ConstantTimeQArray,
    ModulatedTimeQArray,
    PWCTimeQArray,
    SummedTimeQArray,
    TimeQArray,
)

# default total polynomial degree assumed for the operators. Over-declaring is benign
# (the extra monomials are fitted to zero coefficients), under-declaring is caught by
# the residual check in `extend_qarray()`.
DEFAULT_DEGREE = 4

# relative mismatch above which an operator is not the normal-ordered polynomial of the
# declared degree that the estimator assumes. A correct fit sits at ~1e-8 in single
# precision and ~1e-15 in double, a wrong degree at ~1e-1.
_MISMATCH_TOLERANCE = 1e-4

_RCOND = 1e-10


def _ladder_diag(dim: int, p: int, q: int) -> np.ndarray:
    r"""Diagonal of $(a^\dag)^p a^q$ on a Fock space of dimension `dim`.

    Entry `col` is $\bra{col - (q - p)} (a^\dag)^p a^q \ket{col}$, and zero where that
    element falls outside the matrix. Note the only dependence on `dim` is through those
    bounds, which is what makes the diagonal extensible to any dimension.
    """
    delta = q - p
    columns = np.arange(dim, dtype=np.float64)
    out = np.zeros(dim)
    inside = (columns >= q) & (columns - delta >= 0) & (columns - delta < dim)
    n = columns[inside]
    lowered = np.prod([n - i for i in range(q)], axis=0) if q else np.ones_like(n)
    raised = (
        np.prod([n - delta - i for i in range(p)], axis=0) if p else np.ones_like(n)
    )
    out[inside] = np.sqrt(lowered) * np.sqrt(raised)
    return out


@lru_cache
def _mode_map(
    dim: int, extended_dim: int, delta: int, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    r"""One mode's factor of the map extending a diagonal to the enlarged space.

    A normal-ordered polynomial of total degree `degree` puts, on the diagonal of offset
    `delta`, only the monomials $(a^\dag)^{q-\delta} a^q$ with
    `q in [max(0, delta), (degree + delta) // 2]`. Their diagonals are known
    analytically at any dimension, so fitting the truncated diagonal against them and
    re-evaluating at `extended_dim` extends it exactly.

    Returns:
        The extension map, of shape _(extended_dim, dim)_, and the projector onto the
        fitted subspace, of shape _(dim, dim)_, whose distance to the identity on the
        data measures how well the assumption holds.
    """
    if extended_dim == dim:
        identity = np.eye(dim)
        return identity, identity

    qs = range(max(0, delta), (degree + delta) // 2 + 1)
    if len(qs) == 0:
        raise ValueError(
            f'The truncation error estimate cannot represent a diagonal of offset'
            f' {delta} with an assumed polynomial degree of {degree}. Pass a larger'
            f' degree to the `truncation_error` argument of `dq.mesolve()`.'
        )

    truncated = np.stack([_ladder_diag(dim, q - delta, q) for q in qs], axis=-1)
    extended = np.stack([_ladder_diag(extended_dim, q - delta, q) for q in qs], axis=-1)
    # normalise the columns, whose scales otherwise differ by orders of magnitude
    scale = np.linalg.norm(truncated, axis=0)
    scale[scale == 0] = 1.0
    truncated, extended = truncated / scale, extended / scale

    if np.linalg.matrix_rank(truncated) < len(qs):
        raise ValueError(
            f'The truncation error estimate needs at least {len(qs)} usable entries on'
            f' the diagonal of offset {delta} to fit a polynomial of degree {degree},'
            f' but a mode of dimension {dim} does not provide them. Increase the'
            f' truncature, or pass a smaller degree to the `truncation_error` argument'
            f' of `dq.mesolve()`.'
        )

    fit = np.linalg.pinv(truncated, rcond=_RCOND)
    return extended @ fit, truncated @ fit


def _candidate_digits(
    offset: int, dims: tuple[int, ...], degree: int
) -> list[tuple[int, ...]]:
    """Every per-mode split of a flat `dia` offset a polynomial of `degree` can make.

    A monomial shifting mode `k` by `shift_k` is at least of degree `|shift_k|` in that
    mode, so a total degree of `degree` bounds `sum_k |shift_k|`; and a shift beyond
    `dims[k] - 1` falls outside that mode's matrix.
    """
    *slower_dims, dim = dims
    bound = min(degree, dim - 1)
    if not slower_dims:
        return [(offset,)] if abs(offset) <= bound else []

    candidates = []
    residue = offset % dim
    for shift in (residue, residue - dim):
        if abs(shift) > bound:
            continue
        slower = _candidate_digits(
            (offset - shift) // dim, tuple(slower_dims), degree - abs(shift)
        )
        candidates += [(*digits, shift) for digits in slower]
    return candidates


def offset_digits(
    offset: int, dims: tuple[int, ...], degree: int | None = None
) -> tuple[int, ...]:
    """Per-mode shifts of a flat `dia` offset.

    A flat offset is `sum_k shift_k * stride_k`, which the modes after the first share
    modulo their dimension. With `degree` given, the split is the only one a polynomial
    of that degree can make, and an offset admitting several of them is rejected rather
    than guessed. Without it — when the degree is itself derived from the offsets — the
    shifts are the balanced mixed-radix digits, correct as long as
    `|shift_k| <= dims[k] // 2`.
    """
    if degree is None:
        digits = []
        for dim in reversed(dims[1:]):
            residue = offset % dim
            if residue > dim // 2:
                residue -= dim
            digits.append(residue)
            offset = (offset - residue) // dim
        digits.append(offset)
        return tuple(reversed(digits))

    candidates = _candidate_digits(offset, dims, degree)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise ValueError(
            f'The truncation error estimate cannot attribute the diagonal of offset'
            f' {offset} of an operator with dims={dims} to any per-mode shifts of a'
            f' polynomial of degree {degree}. Pass a larger degree to the'
            f' `truncation_error` argument of `dq.mesolve()`.'
        )
    raise ValueError(
        f'The truncation error estimate cannot attribute the diagonal of offset'
        f' {offset} of an operator with dims={dims} unambiguously: a polynomial of'
        f' degree {degree} can put it on the per-mode shifts {candidates[0]} or'
        f' {candidates[1]}. Pass a smaller degree to the `truncation_error` argument of'
        f' `dq.mesolve()`, or increase the truncature of the modes after the first.'
    )


def _leaves(timeqarray: TimeQArray, argname: str) -> list[tuple[Callable, QArray]]:
    """The constant carrier qarrays of a timeqarray, with their scalar prefactor.

    Constant, piecewise-constant and modulated timeqarrays are a fixed qarray times a
    scalar prefactor, so extending them is extending that qarray. A sum applies its own
    prefactor — its clipping — on top of the prefactors of its terms.
    """
    if isinstance(timeqarray, SummedTimeQArray):
        clip = timeqarray._prefactor
        return [
            (lambda t, clip=clip, term=prefactor: clip(t) * term(t), qarray)
            for term in timeqarray.timeqarrays
            for prefactor, qarray in _leaves(term, argname)
        ]
    if isinstance(timeqarray, (ConstantTimeQArray, PWCTimeQArray, ModulatedTimeQArray)):
        return [(timeqarray._prefactor, timeqarray.qarray)]
    raise ValueError(
        f'Argument `{argname}` is a `{type(timeqarray).__name__}`, which the truncation'
        f' error estimate does not support because its extension to a larger Fock space'
        f' cannot be derived. Use `dq.constant()`, `dq.pwc()` or `dq.modulated()`'
        f' instead of `dq.timecallable()`.'
    )


def _leaf_qarrays(timeqarray: TimeQArray, argname: str) -> list[QArray]:
    """The constant carrier qarrays of a timeqarray."""
    return [qarray for _, qarray in _leaves(timeqarray, argname)]


def _dia_data(qarray: QArray, argname: str) -> SparseDIADataArray:
    data = getattr(qarray, 'data', None)
    if qarray.layout is not dia or not isinstance(data, SparseDIADataArray):
        raise ValueError(
            f'Argument `{argname}` must have layout `dia` for the truncation error'
            f' estimate, which derives the extended-space operators from the stored'
            f' diagonals, but has layout `{qarray.layout}`. Build it with'
            f' `layout=dq.dia`, or convert it with `x.assparsedia(offsets=...)`.'
        )
    return data


def _digit_range(
    timeqarray: TimeQArray, argname: str, degree: int | None = None
) -> tuple[list[int], list[int]]:
    """Per-mode smallest and largest offset shift over every diagonal of an operator."""
    dims = timeqarray.dims
    lowest, highest = [0] * len(dims), [0] * len(dims)
    for qarray in _leaf_qarrays(timeqarray, argname):
        for offset in _dia_data(qarray, argname).offsets:
            for mode, shift in enumerate(offset_digits(offset, dims, degree)):
                lowest[mode] = min(lowest[mode], shift)
                highest[mode] = max(highest[mode], shift)
    return lowest, highest


def _named_operators(
    H: TimeQArray, Ls: list[TimeQArray]
) -> list[tuple[str, TimeQArray]]:
    return [('H', H), *[(f'jump_ops[{i}]', L) for i, L in enumerate(Ls)]]


def extension_buffer(
    H: TimeQArray, Ls: list[TimeQArray], degree: int | None = None
) -> tuple[int, ...]:
    r"""Per-mode number of Fock levels the residual can reach past the truncature.

    Writing `P` for the truncated space, `Q` for its complement and `r+`/`r-` for the
    largest raise/lower of an operator in a given mode (an offset lowers when positive):

    - the commutator's `QP` block is $H_{QP}\rho$, so `H` reaches `r+`, which is
      `max|shift|` for a hermitian Hamiltonian;
    - a jump operator needs `L_QP`, reaching `r+`, and $(L^\dag L)_{QP}$, whose column
      `q` sums over rows at most `n-1+r+` and reachable from `q`, hence `r+ + r-`.

    This is tight on every example of the paper: 1 for $H=u(t)(a+a^\dag)$, 2 for
    $L=a^2-\alpha^2$, 4 for the squeezed cat, and $(2, 1)$ for the two-mode
    $H=(a^2-\alpha^2)b^\dag+h.c.$ with $\mathcal{D}_b$.

    A mode of dimension 2 is taken to be a genuine two-level system rather than a
    truncated oscillator, so it gets no buffer: it cannot leak, and $\sigma_-$ is
    indistinguishable from `dq.destroy(2)` on it anyway.
    """
    named = _named_operators(H, Ls)
    for argname, operator in named:
        if operator.dims != H.dims:
            raise ValueError(
                f'Argument `{argname}` must have the same Hilbert space dimensions as'
                f' `H` for the truncation error estimate, but got'
                f' {operator.dims} and {H.dims}.'
            )

    lowest, highest = _digit_range(H, 'H', degree)
    buffer = [max(-low, high) for low, high in zip(lowest, highest, strict=True)]
    for argname, L in named[1:]:
        lowest, highest = _digit_range(L, argname, degree)
        buffer = [
            max(levels, high - low)
            for levels, low, high in zip(buffer, lowest, highest, strict=True)
        ]
    return tuple(
        0 if dim == 2 else levels for dim, levels in zip(H.dims, buffer, strict=True)
    )


def assumed_degree(H: TimeQArray, Ls: list[TimeQArray]) -> int:
    """Total polynomial degree to assume when no degree was requested.

    At least `DEFAULT_DEGREE`, and always large enough for the offsets actually present
    so that a bare power such as `a**6` needs no user input either.
    """
    degree = DEFAULT_DEGREE
    for argname, operator in _named_operators(H, Ls):
        lowest, highest = _digit_range(operator, argname)
        degree = max(degree, *(-x for x in lowest), *highest)
    return degree


def extend_qarray(
    qarray: QArray, extended_dims: tuple[int, ...], degree: int, argname: str
) -> QArray:
    """Extend a normal-ordered polynomial qarray to a larger Fock space.

    Each stored diagonal is fitted against the analytically known diagonals of the
    monomials that can live on it, then re-evaluated on `extended_dims`. The fit is a
    static matrix per mode, so this is jit, vmap and grad compatible.
    """
    dims = qarray.dims
    data = _dia_data(qarray, argname)
    diags = data.diags
    batch_shape = diags.shape[:-2]
    nmodes = len(dims)
    strides = [int(np.prod(extended_dims[mode + 1 :])) for mode in range(nmodes)]
    n_extended = int(np.prod(extended_dims))

    if len(data.offsets) == 0:
        # an operator with no stored diagonal is zero, and stays zero on any space
        zero = jnp.zeros((*batch_shape, n_extended), dtype=diags.dtype)
        return sparsedia_from_dict({0: zero}, dims=extended_dims)

    flatten = lambda x: x.reshape(*batch_shape, -1)
    extended_diags = {}
    mismatch = jnp.zeros(())
    for index, offset in enumerate(data.offsets):
        digits = offset_digits(offset, dims, degree)
        diag = diags[..., index, :].reshape(*batch_shape, *dims)
        block, projected = diag, diag
        for mode in range(nmodes):
            extension, projector = _mode_map(
                dims[mode], extended_dims[mode], digits[mode], degree
            )
            axis = len(batch_shape) + mode
            block = jnp.moveaxis(
                jnp.tensordot(block, jnp.asarray(extension), axes=(axis, 1)), -1, axis
            )
            projected = jnp.moveaxis(
                jnp.tensordot(projected, jnp.asarray(projector), axes=(axis, 1)),
                -1,
                axis,
            )

        norm = jnp.linalg.norm(flatten(diag), axis=-1)
        residual = jnp.linalg.norm(flatten(projected - diag), axis=-1)
        mismatch = jnp.maximum(
            mismatch, jnp.max(residual / jnp.where(norm == 0, 1.0, norm))
        )

        flat = block.reshape(*batch_shape, n_extended)
        extended_offset = sum(d * s for d, s in zip(digits, strides, strict=True))
        keep = (
            slice(extended_offset, None)
            if extended_offset >= 0
            else slice(None, extended_offset)
        )
        extended_diags[extended_offset] = flat[..., keep]

    # `mismatch` is O(1) when the operator is not the normal-ordered polynomial of the
    # declared degree that the extension assumes
    first = next(iter(extended_diags), None)
    if first is not None:
        extended_diags[first] = eqx.error_if(
            extended_diags[first],
            mismatch > _MISMATCH_TOLERANCE,
            f'Argument `{argname}` is not a normal-ordered polynomial of degree'
            f' {degree} in the ladder operators, so the truncation error estimate'
            f' cannot derive it on a larger Fock space. If the operator is a polynomial'
            f' of higher degree, pass that degree to the `truncation_error` argument of'
            f' `dq.mesolve()`.',
        )
    return sparsedia_from_dict(extended_diags, dims=extended_dims)


def extend_timeqarray(
    timeqarray: TimeQArray, extended_dims: tuple[int, ...], degree: int, argname: str
) -> TimeQArray:
    """Extend a timeqarray to a larger Fock space, leaving its time dependence alone."""
    if isinstance(timeqarray, SummedTimeQArray):
        # `replace` rather than a fresh `SummedTimeQArray`, to carry over the sum's own
        # `tstart`/`tend`, which its `__call__` applies on top of the terms
        return replace(
            timeqarray,
            timeqarrays=[
                extend_timeqarray(term, extended_dims, degree, argname)
                for term in timeqarray.timeqarrays
            ],
        )
    qarray = extend_qarray(
        _leaf_qarrays(timeqarray, argname)[0], extended_dims, degree, argname
    )
    return replace(timeqarray, qarray=qarray)


def inner_outer_indices(
    dims: tuple[int, ...], extended_dims: tuple[int, ...]
) -> tuple[Array, Array]:
    """Flat indices of the truncated space and of its complement in the extended space.

    The inner indices come out in the order of the truncated space's own row-major
    flattening, so `rho` can be used against them without any reshaping.
    """
    if len(dims) != len(extended_dims):
        raise ValueError(
            f'Argument `extended_dims` must have the same number of modes as the state,'
            f' but got dims={dims} and extended_dims={extended_dims}.'
        )
    if any(extended < dim for dim, extended in zip(dims, extended_dims, strict=True)):
        raise ValueError(
            f'Argument `extended_dims` must be at least as large as the state'
            f' dimensions in every mode, but got dims={dims} and'
            f' extended_dims={extended_dims}.'
        )

    occupations = np.unravel_index(np.arange(np.prod(extended_dims)), extended_dims)
    inside = np.ones(np.prod(extended_dims), dtype=bool)
    for occupation, dim in zip(occupations, dims, strict=True):
        inside &= occupation < dim
    return jnp.asarray(np.nonzero(inside)[0]), jnp.asarray(np.nonzero(~inside)[0])


class JumpBlocks(NamedTuple):
    """Blocks of one jump operator, as `(prefactor, block)` terms to sum at a time."""

    qp: list[tuple[Callable, Array]]
    pp: list[tuple[Callable, Array]]
    dagger_product_qp: list[tuple[Callable, Array]]


class ResidualBlocks(NamedTuple):
    """Blocks of the extended-space operators the residual is built from."""

    hamiltonian_qp: list[tuple[Callable, Array]]
    jumps: list[JumpBlocks]


def residual_blocks(
    H_extended: TimeQArray, Ls_extended: list[TimeQArray], inner: Array, outer: Array
) -> ResidualBlocks:
    """Blocks of the extended-space operators the truncation error rate needs.

    Only thin `QP` blocks and the `PP` block enter the residual (see
    `truncation_error_rate_of_blocks()`), and the operators are a fixed qarray per term
    times a scalar prefactor, so the blocks are built once here rather than at every
    solver step: what is left to do per step is scaling them by their prefactor.

    Note:
        The `L^dag L` blocks are quadratic in the number of terms of a summed jump
        operator, which is a handful in practice.
    """
    inner_rows, inner_columns = inner[:, None], inner[None, :]
    outer_rows = outer[:, None]

    hamiltonian_qp = [
        (prefactor, qarray.to_jax()[outer_rows, inner_columns])
        for prefactor, qarray in _leaves(H_extended, 'H')
    ]

    jumps = []
    for i, jump_operator in enumerate(Ls_extended):
        terms = [
            (prefactor, qarray.to_jax())
            for prefactor, qarray in _leaves(jump_operator, f'jump_ops[{i}]')
        ]
        # (L^dag L)_QP = sum_jk conj(f_j) f_k L_j[:, Q]^dag L_k[:, P], never forming the
        # full product
        dagger_product_qp: list[tuple[Callable, Array]] = [
            (
                lambda t, left=left, right=right: jnp.conj(left(t)) * right(t),
                jump_left[:, outer].conj().mT @ jump_right[:, inner],
            )
            for left, jump_left in terms
            for right, jump_right in terms
        ]
        jumps.append(
            JumpBlocks(
                qp=[(f, jump[outer_rows, inner_columns]) for f, jump in terms],
                pp=[(f, jump[inner_rows, inner_columns]) for f, jump in terms],
                dagger_product_qp=dagger_product_qp,
            )
        )
    return ResidualBlocks(hamiltonian_qp, jumps)


def _at(terms: list[tuple[Callable, Array]], t: RealScalarLike) -> Array:
    """Sum of `(prefactor, block)` terms, evaluated at time `t`."""
    first, *rest = [
        jnp.asarray(prefactor(t))[..., None, None] * block for prefactor, block in terms
    ]
    return sum(rest, start=first)


def truncation_error_rate_of_blocks(
    t: RealScalarLike, rho: Array, blocks: ResidualBlocks
) -> Array:
    r"""Compute $\| (\mathcal{L} - \mathcal{L}_N)\rho \|_1$, the estimator's integrand.

    Writing `P` for the truncated space and `Q` for its complement in the extended
    space, and using that `rho` lives entirely in `P` and that truncating a
    normal-ordered operator is exactly taking its `PP` block, the inner-space
    Lindbladian cancels and the residual reduces to

        R_PP = -1/2 {M, rho},   M = sum_i L_i_QP^dag L_i_QP
        R_QP = -i H_QP rho + sum_i [L_i_QP rho L_i_PP^dag - 1/2 (L_i^dag L_i)_QP rho]
        R_QQ = sum_i L_i_QP rho L_i_QP^dag
        R_PQ = R_QP^dag

    Every product therefore involves a thin `QP` block, and the extended-space
    Lindbladian is never formed. `R` is moreover low rank, so its trace norm comes from
    a compressed eigenproblem spanned by the range generators of its blocks rather than
    from a dense eigendecomposition of the whole extended space.

    Args:
        t: Time at which to evaluate the operators.
        rho: Density matrix on the truncated space, of shape _(n, n)_.
        blocks: Blocks of the extended-space operators, see `residual_blocks()`.

    Returns:
        The trace norm of the residual, a non-negative scalar.
    """
    # -i H rho, restricted to the rows that leak out of the truncated space
    border = -1j * (_at(blocks.hamiltonian_qp, t) @ rho)

    # per jump operator: the QP block, and `L_QP rho`, which is both a range generator
    # and the factor of every product below
    leaks = []
    for jump in blocks.jumps:
        jump_qp = _at(jump.qp, t)
        leaked = jump_qp @ rho
        border += leaked @ _at(jump.pp, t).conj().mT - 0.5 * (
            _at(jump.dagger_product_qp, t) @ rho
        )
        leaks.append((jump_qp, leaked))

    outer_size, inner_size = border.shape[-2], border.shape[-1]
    corner = jnp.zeros((outer_size, outer_size), dtype=border.dtype)
    for jump_qp, leaked in leaks:
        corner += leaked @ jump_qp.conj().mT

    number_of_generators = outer_size * (1 + 2 * len(leaks))
    if number_of_generators < inner_size:
        # compress the truncated-space block onto the range of the residual
        generators = [border.conj().mT]
        for jump_qp, leaked in leaks:
            generators += [jump_qp.conj().mT, leaked.conj().mT]
        # the generators are rank deficient whenever a `QP` block vanishes (any purely
        # lowering jump operator) or `rho` is close to pure, and the backward pass of a
        # rank-deficient `qr` is `nan`. The basis is only a frame in which to read the
        # residual's spectrum, so freezing it leaves the value untouched and yields the
        # same subgradient as the dense branch (differentiating |eigenvalues| through a
        # fixed frame), instead of poisoning the whole gradient.
        basis, _ = jnp.linalg.qr(jnp.concatenate(generators, axis=-1))
        basis = jax.lax.stop_gradient(basis)
        border = border @ basis
        # basis^dag M rho basis = sum_i (L_QP basis)^dag (L_QP rho basis)
        weighted = jnp.zeros((basis.shape[-1],) * 2, dtype=border.dtype)
        for jump_qp, leaked in leaks:
            weighted += (jump_qp @ basis).conj().mT @ (leaked @ basis)
    else:
        weighted = jnp.zeros_like(rho)
        for jump_qp, leaked in leaks:
            weighted += jump_qp.conj().mT @ leaked

    # R_PP = -1/2 (M rho + rho M) and rho M = (M rho)^dag for hermitian rho and M
    diagonal = -0.5 * (weighted + weighted.conj().mT)

    residual = jnp.concatenate(
        [
            jnp.concatenate([diagonal, border.conj().mT], axis=-1),
            jnp.concatenate([border, corner], axis=-1),
        ],
        axis=-2,
    )
    residual = 0.5 * (residual + residual.conj().mT)
    # the residual is hermitian: its trace norm is the sum of its absolute eigenvalues
    return jnp.abs(jnp.linalg.eigvalsh(residual)).sum(-1)


def truncation_error_rate(
    t: RealScalarLike,
    rho: Array,
    H_extended: TimeQArray,
    Ls_extended: list[TimeQArray],
    inner: Array,
    outer: Array,
) -> Array:
    r"""Compute the estimator's integrand from the extended-space operators.

    A solve builds the blocks once with `residual_blocks()` and calls
    `truncation_error_rate_of_blocks()` at every step; this is the one-shot version.
    """
    blocks = residual_blocks(H_extended, Ls_extended, inner, outer)
    return truncation_error_rate_of_blocks(t, rho, blocks)


def accumulate_truncation_error(step_ts: Array, rates: Array, ts: Array) -> Array:
    """Integrate the estimator's rate over the solver steps and sample it on `ts`.

    Diffrax pads the unused tail of a `steps=True` save buffer with `inf`; those entries
    are collapsed onto the last real step time so that they contribute nothing to the
    cumulative trapezoid.

    Warning:
        The rate is integrated by trapezoid over the solver steps, and `ts` is sampled
        by linear interpolation of the result, so the returned values carry a
        quadrature error of the order of the solver's own error. The bound is rigorous
        up to that error, not beyond it.
    """
    valid = jnp.isfinite(step_ts)
    last = jnp.max(jnp.where(valid, step_ts, -jnp.inf))
    times = jnp.where(valid, step_ts, last)
    rates = jnp.where(valid, rates, 0.0)
    increments = 0.5 * (rates[1:] + rates[:-1]) * jnp.diff(times)
    xi = jnp.concatenate([jnp.zeros((1,), increments.dtype), jnp.cumsum(increments)])
    return jnp.interp(ts, times, xi)
