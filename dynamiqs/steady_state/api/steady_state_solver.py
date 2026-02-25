# The vmap/batching infrastructure and public API design in this module are adapted
# from work by Derek Everett (https://github.com/derekeverett) on the mesteadystate
# solver for dynamiqs.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

import dynamiqs as dq

from ..._checks import check_hermitian, check_qarray_is_dense, check_shape
from ...integrators._utils import cartesian_vmap, catch_xla_runtime_error, multi_vmap
from ...options import Options, check_options
from ...qarrays.qarray import QArray, QArrayLike
from ...qarrays.utils import asqarray
from ..solvers.steady_state_gmres import SteadyStateGMRES

# =============================================================================
# Result types
# =============================================================================


class SteadyStateResult(eqx.Module, ABC):
    """Abstract base class for steady-state solver results."""

    rho: QArray
    infos: eqx.Module

    @staticmethod
    @abstractmethod
    def out_axes(): ...


# =============================================================================
# Solver classes
# =============================================================================


class SteadyStateSolver(eqx.Module, ABC):
    """Abstract base class for steady-state solvers.

    Subclasses must implement `_run` (the core single-instance solver logic)
    and `result_type` (returning the concrete `SteadyStateResult` subclass).
    """

    @abstractmethod
    def _run(
        self, H: QArray, Ls: list[QArray], rho0: QArray | None, options: Options
    ) -> SteadyStateResult: ...

    @staticmethod
    @abstractmethod
    def result_type() -> type[SteadyStateResult]: ...


# =============================================================================
# Public API
# =============================================================================


def steadystate(
    H: QArrayLike,
    jump_ops: list[QArrayLike],
    *,
    rho0: QArrayLike | None = None,
    solver: SteadyStateSolver = SteadyStateGMRES(),  # noqa: B008
    options: Options = Options(),  # noqa: B008
) -> SteadyStateResult:
    r"""Compute the steady state of the Lindblad master equation.

    The Lindblad dynamics are written as
    $$
        \frac{d\rho}{dt} = \mathcal{L}(\rho),
    $$
    with
    $$
        \mathcal{L}(\rho) = -i[H, \rho]
        + \sum_{k=1}^N \left(
            L_k \rho L_k^\dag
            - \frac{1}{2} L_k^\dag L_k \rho
            - \frac{1}{2} \rho L_k^\dag L_k
        \right).
    $$
    This function finds the steady-state density matrix $\rho_\infty$ such that
    $$
        \mathcal{L}(\rho_\infty) = 0.
    $$

    Note:
        This function supports batched computation over `H`, `jump_ops`, and
        `rho0` via JAX's `vmap`, as well as gradient computation.

    Args:
        H *(qarray of shape (..., n, n))*: Hamiltonian.
        jump_ops *(list of qarray, each of shape (..., n, n))*: Jump operators.
        rho0 *(qarray of shape (..., n, n), optional)*: Initial guess for the
            density matrix. Defaults to `None`, which uses the vacuum state
            $|0\rangle\langle 0|$.
        solver: Solver instance controlling the algorithm and its parameters.
            Defaults to `SteadyStateGMRES()`. See `SteadyStateGMRES` for
            available options.
        options: Generic dynamiqs solver options (e.g. `cartesian_batching`).

    Returns:
        `SteadyStateResult` :
            A subclass depending on the solver used. For the
            default `SteadyStateGMRES` solver, returns `SteadyStateGMRESResult`
            with fields:

        **`rho`** *(qarray of shape (..., n, n))* — The steady-state density
          matrix.
        **`infos`** *(`GMRESAuxInfo`)* — Auxiliary solver information
          containing `n_iteration`, `success`, and `recycling`.

    Examples:
        ```python
        import dynamiqs as dq

        n = 16
        a = dq.destroy(n)
        H = a.dag() + a
        jump_ops = [a]

        # With default solver
        result = dq.steadystate(H, jump_ops)
        print(f'Converged: {result.infos.success}')
        print(f'Iterations: {result.infos.n_iteration}')

        # With custom solver parameters
        solver = SteadyStateGMRES(tol=1e-6, krylov_size=64)
        result = dq.steadystate(H, jump_ops, solver=solver)
        ```
    """
    # === convert arguments ===
    H = asqarray(H)
    Ls = [asqarray(L) for L in jump_ops]
    if rho0 is not None:
        rho0 = asqarray(rho0)

    # === check arguments ===
    _check_steadystate_solver(solver)
    _check_steadystate_args(H, Ls, rho0, solver)
    check_options(options, 'steadystate')
    options = options.initialise()

    # === convert rho0 to density matrix ===
    if rho0 is not None:
        rho0 = rho0.todm()
        rho0 = check_hermitian(rho0, 'rho0')

    return _vectorized_steadystate(H, Ls, rho0, solver, options)


# =============================================================================
# Vectorized entry point (shared by all solvers)
# =============================================================================


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'options'))
def _vectorized_steadystate(
    H: QArray,
    Ls: list[QArray],
    rho0: QArray | None,
    solver: SteadyStateSolver,
    options: Options,
) -> SteadyStateResult:
    """Vectorized entry point shared by all steady-state solvers.

    Handles batching (flat or cartesian) via `vmap`, then dispatches to
    ``solver._run`` for each individual (non-batched) solve.
    """
    if options is None:
        options = Options(cartesian_batching=False)

    def _qarray_in_axes(q: QArray) -> int | None:
        return 0 if q.ndim > 2 else None

    # --- build in_axes and out_axes ---
    H_in_axes = _qarray_in_axes(H)
    Ls_in_axes = [_qarray_in_axes(L) for L in Ls]
    rho0_in_axes = _qarray_in_axes(rho0) if rho0 is not None else None
    in_axes = (H_in_axes, Ls_in_axes, rho0_in_axes, None)

    out_axes = solver.result_type().out_axes()

    # Closure that captures `solver` (static) and calls its _run method.
    def _run_single(
        H: QArray, Ls: list[QArray], rho0: QArray | None, options: Options
    ) -> SteadyStateResult:
        return solver._run(H, Ls, rho0, options)

    # --- cartesian batching ---
    if options.cartesian_batching:
        rho0_nvmap = rho0.ndim - 2 if rho0 is not None else 0
        nvmap = (H.ndim - 2, [L.ndim - 2 for L in Ls], rho0_nvmap, 0)
        f = cartesian_vmap(_run_single, in_axes, out_axes, nvmap)

    # --- flat (broadcast) batching ---
    else:
        arrays_to_broadcast = [H, *Ls]
        if rho0 is not None:
            arrays_to_broadcast.append(rho0)
        bshape = jnp.broadcast_shapes(*[x.shape[:-2] for x in arrays_to_broadcast])
        nvmap = len(bshape)

        n = H.shape[-1]
        H = H.broadcast_to(*bshape, n, n)
        Ls = [L.broadcast_to(*bshape, n, n) for L in Ls]
        if rho0 is not None:
            rho0 = rho0.broadcast_to(*bshape, n, n)

        f = multi_vmap(_run_single, in_axes, out_axes, nvmap)

    return f(H, Ls, rho0, options)


# =============================================================================
# Argument checking
# =============================================================================


def _check_steadystate_solver(solver: SteadyStateSolver):
    if not isinstance(solver, SteadyStateSolver):
        raise TypeError(
            f'Argument `solver` must be an instance of `SteadyStateSolver`, '
            f'got {type(solver).__name__}.'
        )


def _check_steadystate_args(
    H: QArray, Ls: list[QArray], rho0: QArray | None, solver: SteadyStateSolver
):
    # === check H shape ===
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check Ls shape ===
    for i, L in enumerate(Ls):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})

    if len(Ls) == 0:
        warnings.warn(
            'Argument `jump_ops` is an empty list. The Lindbladian reduces to '
            'unitary evolution, which has no unique steady state unless the '
            'Hamiltonian is zero.',
            stacklevel=3,
        )

    # === check rho0 shape and layout ===
    if rho0 is not None:
        check_shape(rho0, 'rho0', '(..., n, 1)', '(..., n, n)', subs={'...': '...rho0'})
        check_qarray_is_dense(rho0, 'rho0')

    should_check = getattr(solver, 'check_pure_imaginary_eigenvalue', False)
    if should_check:
        _check_G_has_no_pure_imaginary_eigenvalue(H, Ls)


def _check_G_has_no_pure_imaginary_eigenvalue(H: QArray, Ls: list[QArray]) -> None:
    H_mat = H.to_jax()
    if len(Ls) == 0:
        LdagL = jnp.zeros_like(H_mat)
    else:
        Ls_q = dq.stack(Ls)
        LdagL = (Ls_q.dag() @ Ls_q).sum(0).to_jax()

    G = -1j * H_mat - 0.5 * LdagL
    eigvals = jnp.linalg.eigvals(G)

    real_dtype = jnp.real(jnp.zeros((), dtype=eigvals.dtype)).dtype
    atol = float(jnp.finfo(real_dtype).eps)

    has_pure_imag = jnp.any(
        jnp.isclose(jnp.real(eigvals), 0.0, rtol=0.0, atol=atol), axis=-1
    )
    any_batch_has_one = jnp.any(has_pure_imag)
    if bool(jax.device_get(any_batch_has_one)):
        raise ValueError(
            'G has at least one purely imaginary eigenvalue within machine '
            f'precision (atol={atol:.3e}). This can make GMRES fail.'
        )
