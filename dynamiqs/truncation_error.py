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

from dataclasses import replace
from functools import lru_cache

import equinox as eqx
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


def offset_digits(offset: int, dims: tuple[int, ...]) -> tuple[int, ...]:
    """Per-mode shifts of a flat `dia` offset, as balanced mixed-radix digits.

    A flat offset is `sum_k shift_k * stride_k`. The slowest mode has no wrap-around
    above it so its shift is exact; the others are recovered as balanced residues, which
    is the right split as long as `|shift_k| <= dims[k] // 2`. A wrong split lands the
    diagonal on monomials it cannot be fitted against, which `extend_qarray()` reports.
    """
    digits = []
    for dim in reversed(dims[1:]):
        residue = offset % dim
        if residue > dim // 2:
            residue -= dim
        digits.append(residue)
        offset = (offset - residue) // dim
    digits.append(offset)
    return tuple(reversed(digits))


def _leaf_qarrays(timeqarray: TimeQArray, argname: str) -> list[QArray]:
    """The constant carrier qarrays of a timeqarray.

    Constant, piecewise-constant and modulated timeqarrays are a fixed qarray times a
    scalar prefactor, so extending them is extending that qarray.
    """
    if isinstance(timeqarray, SummedTimeQArray):
        return [
            qarray
            for term in timeqarray.timeqarrays
            for qarray in _leaf_qarrays(term, argname)
        ]
    if isinstance(timeqarray, (ConstantTimeQArray, PWCTimeQArray, ModulatedTimeQArray)):
        return [timeqarray.qarray]
    raise ValueError(
        f'Argument `{argname}` is a `{type(timeqarray).__name__}`, which the truncation'
        f' error estimate does not support because its extension to a larger Fock space'
        f' cannot be derived. Use `dq.constant()`, `dq.pwc()` or `dq.modulated()`'
        f' instead of `dq.timecallable()`.'
    )


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


def _digit_range(timeqarray: TimeQArray, argname: str) -> tuple[list[int], list[int]]:
    """Per-mode smallest and largest offset shift over every diagonal of an operator."""
    dims = timeqarray.dims
    lowest, highest = [0] * len(dims), [0] * len(dims)
    for qarray in _leaf_qarrays(timeqarray, argname):
        for offset in _dia_data(qarray, argname).offsets:
            for mode, shift in enumerate(offset_digits(offset, dims)):
                lowest[mode] = min(lowest[mode], shift)
                highest[mode] = max(highest[mode], shift)
    return lowest, highest


def _named_operators(
    H: TimeQArray, Ls: list[TimeQArray]
) -> list[tuple[str, TimeQArray]]:
    return [('H', H), *[(f'jump_ops[{i}]', L) for i, L in enumerate(Ls)]]


def extension_buffer(H: TimeQArray, Ls: list[TimeQArray]) -> tuple[int, ...]:
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
    """
    for argname, operator in _named_operators(H, Ls):
        if operator.dims != H.dims:
            raise ValueError(
                f'Argument `{argname}` must have the same Hilbert space dimensions as'
                f' `H` for the truncation error estimate, but got'
                f' {operator.dims} and {H.dims}.'
            )

    lowest, highest = _digit_range(H, 'H')
    buffer = [max(-low, high) for low, high in zip(lowest, highest, strict=True)]
    for i, L in enumerate(Ls):
        lowest, highest = _digit_range(L, f'jump_ops[{i}]')
        buffer = [
            max(levels, high - low)
            for levels, low, high in zip(buffer, lowest, highest, strict=True)
        ]
    return tuple(buffer)


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

    flatten = lambda x: x.reshape(*batch_shape, -1)
    extended_diags = {}
    mismatch = jnp.zeros(())
    for index, offset in enumerate(data.offsets):
        digits = offset_digits(offset, dims)
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


def truncation_error_rate(
    t: RealScalarLike,
    rho: Array,
    H_extended: TimeQArray,
    Ls_extended: list[TimeQArray],
    inner: Array,
    outer: Array,
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
        H_extended: Hamiltonian on the extended space.
        Ls_extended: Jump operators on the extended space.
        inner: Flat extended-space indices of the truncated space.
        outer: Flat extended-space indices of its complement.

    Returns:
        The trace norm of the residual, a non-negative scalar.
    """
    inner_rows, inner_columns = inner[:, None], inner[None, :]
    outer_rows = outer[:, None]

    hamiltonian = H_extended(t).to_jax()
    # -i H rho, restricted to the rows that leak out of the truncated space
    border = -1j * (hamiltonian[outer_rows, inner_columns] @ rho)

    # per jump operator: the QP block, the PP block, the QP block of L^dag L, and
    # `L_QP rho`, which is both a range generator and the factor of every product below
    blocks = []
    for jump_operator in Ls_extended:
        jump = jump_operator(t).to_jax()
        jump_qp = jump[outer_rows, inner_columns]
        jump_pp = jump[inner_rows, inner_columns]
        # (L^dag L)_QP = L[:, Q]^dag L[:, P], never forming the full product
        dagger_product_qp = jump[:, outer].conj().mT @ jump[:, inner]
        blocks.append((jump_qp, jump_pp, dagger_product_qp, jump_qp @ rho))

    for _, jump_pp, dagger_product_qp, leaked in blocks:
        border += leaked @ jump_pp.conj().mT - 0.5 * (dagger_product_qp @ rho)

    corner = jnp.zeros((outer.size, outer.size), dtype=border.dtype)
    for jump_qp, _, _, leaked in blocks:
        corner += leaked @ jump_qp.conj().mT

    number_of_generators = outer.size * (1 + 2 * len(blocks))
    if number_of_generators < inner.size:
        # compress the truncated-space block onto the range of the residual
        generators = [border.conj().mT]
        for jump_qp, _, _, leaked in blocks:
            generators += [jump_qp.conj().mT, leaked.conj().mT]
        basis, _ = jnp.linalg.qr(jnp.concatenate(generators, axis=-1))
        border = border @ basis
        # basis^dag M rho basis = sum_i (L_QP basis)^dag (L_QP rho basis)
        weighted = jnp.zeros((basis.shape[-1],) * 2, dtype=border.dtype)
        for jump_qp, _, _, leaked in blocks:
            weighted += (jump_qp @ basis).conj().mT @ (leaked @ basis)
    else:
        weighted = jnp.zeros_like(rho)
        for jump_qp, _, _, leaked in blocks:
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


def accumulate_truncation_error(step_ts: Array, rates: Array, ts: Array) -> Array:
    """Integrate the estimator's rate over the solver steps and sample it on `ts`.

    Diffrax pads the unused tail of a `steps=True` save buffer with `inf`; those entries
    are collapsed onto the last real step time so that they contribute nothing to the
    cumulative trapezoid.

    Warning:
        `ts` is sampled by linear interpolation of a quantity that is only piecewise
        linear between solver steps, so the values in between step times carry a small
        quadrature error.
    """
    valid = jnp.isfinite(step_ts)
    last = jnp.max(jnp.where(valid, step_ts, -jnp.inf))
    times = jnp.where(valid, step_ts, last)
    rates = jnp.where(valid, rates, 0.0)
    increments = 0.5 * (rates[1:] + rates[:-1]) * jnp.diff(times)
    xi = jnp.concatenate([jnp.zeros((1,), increments.dtype), jnp.cumsum(increments)])
    return jnp.interp(ts, times, xi)
