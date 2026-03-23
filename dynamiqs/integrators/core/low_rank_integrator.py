from __future__ import annotations

import warnings
from functools import partial

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array
from jaxtyping import PyTree

from ..._checks import check_hermitian
from ...method import Dopri5, Dopri8, Euler, Kvaerno3, Kvaerno5, LinearSolver, Tsit5
from ...qarrays.utils import asqarray
from ...result import MESolveLowRankResult, Result, Saved, SolveSaved
from .._utils import assert_method_supported
from .abstract_integrator import BaseIntegrator
from .diffrax_integrator import AdaptiveStepInfos, FixedStepInfos, call_diffeqsolve
from .interfaces import MEInterface, SolveInterface
from .save_mixin import SolveSaveMixin


def qr_solve(A: Array, B: Array) -> Array:
    operator = lx.MatrixLinearOperator(A)

    def solve_single(b: Array) -> Array:
        return lx.linear_solve(operator, b, throw=False, solver=lx.QR()).value

    return jax.vmap(solve_single, in_axes=1, out_axes=1)(B)  # vmap over columns of B


def cholesky_solve(A: Array, B: Array) -> Array:
    A_pinv = jax.scipy.linalg.solve(A.conj().T @ A, A.conj().T, assume_a='pos')
    return A_pinv @ B


def normalize_m(m: Array) -> Array:
    norm = jnp.sqrt(jnp.sum(jnp.abs(m) ** 2, axis=(-2, -1), keepdims=True))
    return m / norm


def rho_from_m(m: Array) -> Array:
    rho = m @ m.conj().swapaxes(-2, -1)
    tr = jnp.trace(rho, axis1=-2, axis2=-1)
    return rho / tr[..., None, None]


def initialize_m0_from_ket(psi0: Array, rank: int, eps: float, key: Array) -> Array:
    """Initialize the low-rank d.m. m0 from a pure state |psi0>.

    The first column of m0 is (a rescaled) psi0, and the remaining rank-1 columns
    are small random vectors orthogonal to psi0, scaled by `eps`.
    This ensures the initial density matrix rho0 = m0 @ m0† ≈ |psi0><psi0|
    while making the target rank for numerical stability.
    """
    # psi0 has shape (n, 1)
    n = psi0.shape[0]
    psi0 = psi0 / jnp.linalg.norm(psi0)
    psi0_unit = psi0

    # rescale the leading column so that its weight plus the rank-1 perturbation
    # columns (each with norm eps) sum to 1 in the trace of rho.
    if rank > 1:
        psi0 = psi0_unit * jnp.sqrt(jnp.maximum(1.0 - (rank - 1) * (eps**2), 0.0))

    if rank == 1:
        return normalize_m(psi0)

    # generate random complex vectors for the remaining rank-1 columns.
    key_r, key_i = jax.random.split(key)
    rand_r = jax.random.normal(key_r, (n, rank - 1), dtype=psi0.real.dtype)
    rand_i = jax.random.normal(key_i, (n, rank - 1), dtype=psi0.real.dtype)
    rand = (rand_r + 1j * rand_i) / jnp.sqrt(2.0)

    # project out the psi0 component to make perturbations orthogonal to psi0,
    # then QR-orthonormalize them before scaling by eps.
    rand = rand - psi0_unit @ (psi0_unit.conj().T @ rand)
    q, _ = jnp.linalg.qr(rand, mode='reduced')
    m0 = jnp.concatenate([psi0, q * eps], axis=1)
    return normalize_m(m0)


def initialize_m0_from_dm(
    rho0: Array, rank: int, eps: float, key: Array, eigval_tol: float = 1e-12
) -> Array:
    """Initialize the low-rank d.m. m0 from a density matrix rho0.

    The largest `rank` eigenvectors of rho0 are used as columns of m0, scaled by
    the square root of their eigenvalues. Columns corresponding to
    near-zero eigenvalues (below `eigval_tol`) are replaced by small random
    perturbations scaled by `eps`, ensuring that the rank is at least `rank`.
    """
    # eigendecompose rho0 and keep the `rank` largest eigenvalues/vectors.
    evals, evecs = jnp.linalg.eigh(rho0)
    evals = jnp.maximum(evals, 0.0)
    evals_rank = evals[-rank:][::-1]
    cols = evecs[:, -rank:][:, ::-1] * jnp.sqrt(evals_rank)[None, :]

    # for columns with near-zero eigenvalues, add small random perturbations
    # scaled by eps to avoid degeneracies during solve.
    key_r, key_i = jax.random.split(key)
    rand_r = jax.random.normal(key_r, (rho0.shape[0], rank), dtype=cols.real.dtype)
    rand_i = jax.random.normal(key_i, (rho0.shape[0], rank), dtype=cols.real.dtype)
    rand = (rand_r + 1j * rand_i) / jnp.sqrt(2.0)

    # Only perturb columns whose eigenvalue is below the tolerance.
    tol = eigval_tol * jnp.maximum(jnp.max(evals_rank), 1.0)
    mask = (evals_rank <= tol).astype(cols.real.dtype)
    cols = cols + rand * (eps * mask)[None, :]

    return normalize_m(cols)


def expval_from_m(m: Array, op: Array) -> Array:
    return jnp.sum(jnp.conj(m) * (op @ m))


class MESolveLowRankIntegrator(
    BaseIntegrator, MEInterface, SolveSaveMixin, SolveInterface
):
    """Implementation of low-rank method from Goutte, Savona (2025) arxiv:2508.18114.

    Original implementation in Julia:
    https://github.com/leogoutte/low_rank/blob/main/src/low_rank.jl
    """

    @property
    def dims(self) -> tuple[int, ...]:
        return self.H.dims

    @property
    def Es_jax(self) -> list[Array] | None:
        return [E.to_jax() for E in self.Es] if self.Es is not None else None

    def __post_init__(self):
        # check that assume_hermitian is True (required for the low-rank solver)
        if not self.options.assume_hermitian:
            raise ValueError(
                'The LowRank method requires `dq.Options(assume_hermitian=True, ...)`'
                '(default value). '
                'If you initial state is not Hermitian, consider using other methods.'
            )

        # check hermiticity for density matrix inputs
        if self.y0.isdm():
            self.y0 = check_hermitian(self.y0, 'rho0')

        # check that the rank is smaller than the initial state size
        n = self.y0.shape[-2]
        if n < self.method.rank:
            raise ValueError(
                'Argument `rank` must be smaller than the initial state size `n`,'
                f'but got rank={self.method.rank} and n={n}.'
            )

        # warn if using Cholesky solver with single precision
        if self.method.linear_solver is LinearSolver.CHOLESKY and not jax.config.read(
            'jax_enable_x64'
        ):
            warnings.warn(
                'Using the Cholesky linear solver with single-precision dtypes can be '
                'numerically unstable; consider enabling double precision with '
                "`dq.set_precision('double')`.",
                stacklevel=2,
            )

        # check internal ODE solver
        supported_ode_methods = (Euler, Dopri5, Dopri8, Tsit5, Kvaerno3, Kvaerno5)
        assert_method_supported(self.method.ode_method, supported_ode_methods)
        self.method.ode_method.assert_supports_gradient(self.gradient)

    def run(self) -> Result:
        # initialize low-rank representation from ket or density matrix input
        eps = self.method.perturbation_scale
        if self.y0.isket():
            psi0 = self.y0.to_jax()
            m0 = initialize_m0_from_ket(psi0, self.method.rank, eps, self.method.key)
        else:
            rho0 = self.y0.todm().to_jax()
            m0 = initialize_m0_from_dm(rho0, self.method.rank, eps, self.method.key)

        # define diffrax term
        linear_solvers = {
            LinearSolver.CHOLESKY: cholesky_solve,
            LinearSolver.QR: qr_solve,
        }
        linsolve = linear_solvers[self.method.linear_solver]

        def vector_field(t, m, _):  # noqa: ANN001, ANN202
            H = self.H(t)
            Ls = [L(t) for L in self.Ls]
            dm = (-1j) * (H @ m).to_jax()

            for L in Ls:
                Lm = (L @ m).to_jax()
                tmp = linsolve(m, Lm)
                dm += 0.5 * (Lm @ tmp.conj().T) - 0.5 * (L.dag() @ Lm).to_jax()

            return dm

        term = dx.ODETerm(vector_field)

        # call diffrax to solve the ODE and save the results
        solution = call_diffeqsolve(
            self.ts,
            m0,
            term,
            self.method.ode_method,
            self.gradient,
            self.options,
            self.discontinuity_ts,
            save=self.save,
        )

        saved = self.postprocess_saved(*solution.ys)
        return self.result(saved, infos=self.infos(solution.stats))

    def save(self, m: PyTree) -> SolveSaved:
        m = normalize_m(m)

        msave = None
        if self.options.save_states:
            msave = asqarray(m, dims=self.dims)

        extra = None
        if self.options.save_extra:
            rho = asqarray(rho_from_m(m), dims=self.dims)
            extra = self.options.save_extra(rho)

        if self.Es_jax is not None:
            Esave = jnp.stack([expval_from_m(m, E) for E in self.Es_jax])
        else:
            Esave = None

        return SolveSaved(ysave=msave, extra=extra, Esave=Esave)

    def postprocess_saved(self, saved: Saved, mlast: PyTree) -> Saved:
        if not self.options.save_states:
            mlast_save = asqarray(normalize_m(mlast), dims=self.dims)
            saved = eqx.tree_at(
                lambda x: x.ysave, saved, mlast_save, is_leaf=lambda x: x is None
            )

        return self.reorder_Esave(saved)

    def infos(self, stats: dict[str, Array]) -> PyTree:
        fixed_step = {
            Euler: True,
            Dopri5: False,
            Dopri8: False,
            Tsit5: False,
            Kvaerno3: False,
            Kvaerno5: False,
        }[type(self.method.ode_method)]

        if fixed_step:
            return FixedStepInfos(stats['num_steps'])
        else:
            return AdaptiveStepInfos(
                stats['num_steps'],
                stats['num_accepted_steps'],
                stats['num_rejected_steps'],
            )


mesolve_lowrank_integrator_constructor = partial(
    MESolveLowRankIntegrator, result_class=MESolveLowRankResult
)
