from __future__ import annotations

import warnings
from dataclasses import replace

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array
from jaxtyping import PyTree

from ..._checks import check_hermitian
from ...method import Dopri5, Dopri8, Euler, Kvaerno3, Kvaerno5, LowRank, Tsit5
from ...qarrays.utils import asqarray
from ...result import MESolveLowRankResult, Saved, SolveSaved
from .._utils import assert_method_supported
from .diffrax_integrator import DiffraxIntegrator
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


def initialize_m0_from_ket(psi0: Array, rank: int, *, eps: float, key: Array) -> Array:
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

    # Rescale the leading column so that its weight plus the rank-1 perturbation
    # columns (each with norm eps) sum to 1 in the trace of rho.
    if rank > 1:
        psi0 = psi0_unit * jnp.sqrt(jnp.maximum(1.0 - (rank - 1) * (eps**2), 0.0))

    if rank == 1:
        return normalize_m(psi0)

    # Generate random complex vectors for the remaining rank-1 columns.
    key_r, key_i = jax.random.split(key)
    rand_r = jax.random.normal(key_r, (n, rank - 1), dtype=psi0.real.dtype)
    rand_i = jax.random.normal(key_i, (n, rank - 1), dtype=psi0.real.dtype)
    rand = (rand_r + 1j * rand_i) / jnp.sqrt(2.0)

    # Project out the psi0 component to make perturbations orthogonal to psi0,
    # then QR-orthonormalize them before scaling by eps.
    rand = rand - psi0_unit @ (psi0_unit.conj().T @ rand)
    q, _ = jnp.linalg.qr(rand, mode='reduced')
    m0 = jnp.concatenate([psi0, q * eps], axis=1)
    return normalize_m(m0)


def initialize_m0_from_dm(
    rho0: Array, rank: int, *, eps: float, key: Array, eigval_tol: float = 1e-12
) -> Array:
    """Initialize the low-rank d.m. m0 from a density matrix rho0.

    The largest `rank` eigenvectors of rho0 are used as columns of m0, scaled by
    the square root of their eigenvalues. Columns corresponding to
    near-zero eigenvalues (below `eigval_tol`) are replaced by small random
    perturbations scaled by `eps`, ensuring that the rank is at least `rank`.
    """
    # Eigendecompose rho0 and keep the `rank` largest eigenvalues/vectors.
    evals, evecs = jnp.linalg.eigh(rho0)
    evals = jnp.maximum(evals, 0.0)
    evals_rank = evals[-rank:][::-1]
    cols = evecs[:, -rank:][:, ::-1] * jnp.sqrt(evals_rank)[None, :]

    # For columns with near-zero eigenvalues, add small random perturbations
    # scaled by eps to avoid degeneracies during solve.
    if eps > 0.0:
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
    DiffraxIntegrator, MEInterface, SolveSaveMixin, SolveInterface
):
    """Implementation of low-rank method from Goutte, Savona (2025) arxiv:2508.18114.

    Original implementation in Julia:
    https://github.com/leogoutte/low_rank/blob/main/src/low_rank.jl
    """

    method: LowRank
    diffrax_solver: dx.AbstractSolver = eqx.field(static=True, default=dx.Tsit5())
    fixed_step: bool = eqx.field(static=True, default=False)
    linear_solver: str = eqx.field(static=True, default='qr')
    dims: tuple[int, ...] | None = eqx.field(static=True, default=None)
    Es_jax: list[Array] | None = None

    def __post_init__(self):
        # check that assume_hermitian is True (required for the low-rank solver)
        if not self.options.assume_hermitian:
            raise ValueError(
                'The low-rank solver requires `assume_hermitian=True` (the default). '
                'Set `options=dq.Options(assume_hermitian=True)` or omit the option.'
            )

        # check hermiticity for density matrix inputs
        if self.y0.isdm():
            self.y0 = check_hermitian(self.y0, 'rho0')

        n = self.y0.shape[-2]
        if n < self.method.rank:
            raise ValueError(
                f'Argument `rank` must be <= n, but is rank={self.method.rank} (n={n}).'
            )
        if self.method.linear_solver == 'cholesky' and not jax.config.read(
            'jax_enable_x64'
        ):
            warnings.warn(
                'Using the Cholesky linear solver with single-precision dtypes can be '
                'numerically unstable; consider enabling double precision with '
                "`dq.set_precision('double')`.",
                stacklevel=2,
            )

        # initialize low-rank representation from ket or density matrix input
        self.dims = self.y0.dims
        eps = self.method.init_perturbation_scale
        if self.y0.isket():
            psi0 = self.y0.to_jax()
            self.y0 = initialize_m0_from_ket(
                psi0, self.method.rank, eps=eps, key=self.method.key
            )
        else:
            rho0_dm = self.y0.todm()
            self.y0 = initialize_m0_from_dm(
                rho0_dm.to_jax(), self.method.rank, eps=eps, key=self.method.key
            )

        self.Es_jax = [E.to_jax() for E in self.Es] if self.Es is not None else None

        # select low-rank inner ODE method/solver
        supported_ode_methods = (Euler, Dopri5, Dopri8, Tsit5, Kvaerno3, Kvaerno5)
        ode_method_constructors = {
            Euler: (dx.Euler(), True),
            Dopri5: (dx.Dopri5(), False),
            Dopri8: (dx.Dopri8(), False),
            Tsit5: (dx.Tsit5(), False),
            Kvaerno3: (dx.Kvaerno3(), False),
            Kvaerno5: (dx.Kvaerno5(), False),
        }
        assert_method_supported(self.method.ode_method, supported_ode_methods)
        ode_method = self.method.ode_method
        self.diffrax_solver, self.fixed_step = ode_method_constructors[type(ode_method)]
        ode_method.assert_supports_gradient(self.gradient)

        # save low-rank-specific settings and result container
        self.linear_solver = self.method.linear_solver
        self.result_class = MESolveLowRankResult

    # we need to redefine these since the Diffrax method is now store in ode_method
    # We could avoid this by defining a _method() property in DiffraxIntegrator and
    # overriding it here to return self.ode_method, but this requires changes
    # outside this file.
    @property
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        ode_method = self.method.ode_method
        if self.fixed_step:
            return dx.ConstantStepSize()
        jump_ts = None if len(self.discontinuity_ts) == 0 else self.discontinuity_ts
        controller = dx.PIDController(
            rtol=ode_method.rtol,
            atol=ode_method.atol,
            safety=ode_method.safety_factor,
            factormin=ode_method.min_factor,
            factormax=ode_method.max_factor,
        )
        if jump_ts is not None:
            controller = replace(controller, jump_ts=jump_ts)
        return controller

    @property
    def dt0(self) -> float | None:
        ode_method = self.method.ode_method
        return ode_method.dt if self.fixed_step else None

    @property
    def max_steps(self) -> int:
        # TODO: fix hard-coded max_steps for fixed methods
        ode_method = self.method.ode_method
        return 100_000 if self.fixed_step else ode_method.max_steps

    @property
    def terms(self) -> dx.AbstractTerm:
        solve = cholesky_solve if self.linear_solver == 'cholesky' else qr_solve

        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            m = y

            H = self.H(t)
            Ls = [L(t) for L in self.Ls]
            dm = (-1j) * (H @ m).to_jax()

            if len(Ls) > 0:
                for L in Ls:
                    Lm = (L @ m).to_jax()
                    tmp = solve(m, Lm)
                    dm = dm + 0.5 * (Lm @ tmp.conj().T) - 0.5 * (L.dag() @ Lm).to_jax()

            return dm

        return dx.ODETerm(vector_field)

    def _rho_from_m(self, m: Array):  # noqa: ANN202
        return asqarray(rho_from_m(m), dims=self.dims)

    def save(self, y: PyTree) -> SolveSaved:
        m = normalize_m(y)
        rho = self._rho_from_m(m) if self.options.save_extra is not None else None

        ysave = None
        if self.options.save_states:
            ysave = asqarray(m, dims=self.dims)
        extra = self.options.save_extra(rho) if self.options.save_extra else None

        if self.Es_jax is not None:
            Esave = jnp.stack([expval_from_m(m, E) for E in self.Es_jax])
        else:
            Esave = None

        return SolveSaved(ysave=ysave, extra=extra, Esave=Esave)

    def postprocess_saved(self, saved: Saved, ylast: PyTree) -> Saved:
        if not self.options.save_states:
            mlast = normalize_m(ylast)
            ylast_save = asqarray(mlast, dims=self.dims)
            saved = eqx.tree_at(
                lambda x: x.ysave, saved, ylast_save, is_leaf=lambda x: x is None
            )

        return self.reorder_Esave(saved)


mesolve_lowrank_integrator_constructor = MESolveLowRankIntegrator
