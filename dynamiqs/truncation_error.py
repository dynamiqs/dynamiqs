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
extended by a finite per-mode buffer, so it can be computed from a second compilation of
the same operators on that extended space.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from diffrax._custom_types import RealScalarLike
from jax import Array

from .time_qarray import TimeQArray

__all__ = ['TruncationError']


class TruncationError(eqx.Module):
    """Extended-space operators for the a posteriori Fock truncation error estimate.

    Pass an instance to the `truncation_error` argument of
    [`dq.mesolve()`][dynamiqs.mesolve] to have the estimator integrated alongside the
    master equation, and read the result from `result.truncation_error`.

    The operators must be the *same* symbolic Hamiltonian and jump operators as the ones
    being solved, compiled on a Fock space enlarged by enough levels per mode that the
    residual is exactly representable there. Sizing that buffer is the caller's job: for
    a monomial of net raise `r` (creations minus annihilations) the Hamiltonian needs
    `max(r, -r)` levels and a jump operator needs `max(r+, r+ - r-)`, taken per mode
    over the polynomial's monomials.

    Warning:
        The operators are assumed to be normal-ordered, so that truncating them commutes
        with taking the inner block of their extended representation.

    The extended Fock dimensions are read from `H.dims`, and the jump operators must be
    given in the same order as the ones being solved.

    Attributes:
        H: Hamiltonian on the extended space, of shape _(..., N, N)_.
        Ls: Jump operators on the extended space, each of shape _(..., N, N)_.
    """

    H: TimeQArray
    Ls: list[TimeQArray]

    @property
    def extended_dims(self) -> tuple[int, ...]:
        return self.H.dims


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
