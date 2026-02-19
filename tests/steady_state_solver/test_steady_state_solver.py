"""Tests for the steadystate solver.

Tests convergence at different tolerances and system sizes for:
- Two-mode systems (build_two_modes with kappa_a=1)
- Random single-mode systems (build_random_single_mode with gamma=1 and gamma=0.1)
Also tests JIT compatibility.
"""

import jax
import jax.numpy as jnp
import dynamiqs as dq
import pytest

from dynamiqs.steady_state import (
    SteadyStateGMRES,
    SteadyStateResult,
    SteadyStateGMRESResult,
)
from .systems import build_two_modes, build_random_single_mode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[12, 24, 32], ids=lambda na: f'na={na}')
def na(request):
    return request.param


@pytest.fixture(
    params=[('single', False), ('double', True)], ids=lambda p: f'prec={p[0]}'
)
def precision(request):
    mode, enable_x64 = request.param
    jax.config.update('jax_enable_x64', enable_x64)
    return mode


@pytest.fixture
def nb(na):
    return na // 4


@pytest.fixture(params=[0, 1], ids=lambda i: f'tol_idx={i}')
def tol(request, precision):
    if precision == 'single':
        tols = (1e-1, 1e-4)
    else:
        tols = (1e-4, 1e-8)
    return tols[request.param]


@pytest.fixture(params=[1.0, 0.1], ids=lambda g: f'gamma={g}')
def gamma(request):
    return request.param


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def lindbladian_residual(H, Ls, rho):
    """Compute max-abs norm of L(rho), i.e. max|lindbladian(rho)|."""
    L_rho = dq.lindbladian(H, Ls, rho)
    return float(jnp.max(jnp.abs(L_rho.to_jax())))


def _dm_atol(precision):
    """Return the absolute tolerance for density matrix checks."""
    return 1e-8 if precision == 'double' else 1e-4


def _simple_oscillator_analytical_steady_state(n, epsilon_a, kappa):
    """Analytical steady state for H=epsilon_a*(a+a^dagger), L=sqrt(kappa)*a."""
    alpha_ss = -2j * epsilon_a / kappa
    return dq.coherent_dm(n, alpha_ss)


def assert_valid_dm(rho, precision):
    """Check that rho is a valid density matrix (hermitian, trace 1, PSD)."""
    atol = _dm_atol(precision)
    rho_jax = rho.to_jax()
    assert jnp.allclose(rho_jax, rho_jax.conj().T, atol=atol), 'rho is not Hermitian'
    assert jnp.isclose(jnp.trace(rho_jax), 1.0, atol=atol), (
        f'Tr(rho) = {jnp.trace(rho_jax)}, expected 1.0'
    )
    eigvals = jnp.linalg.eigvalsh(rho_jax)
    assert jnp.all(eigvals > -atol), (
        f'rho has negative eigenvalue: {float(jnp.min(eigvals))}'
    )


# ---------------------------------------------------------------------------
# Two-mode tests
# ---------------------------------------------------------------------------


class TestTwoModes:
    """Steady-state solver on the two-mode cat-qubit system."""

    def test_convergence(self, na, nb, tol, precision):
        """Solver converges and residual is below tolerance."""
        H, Ls = build_two_modes(na, nb, kappa_a=1)
        solver = SteadyStateGMRES(
            tol=tol, max_iteration=200, krylov_size=64, exact_dm=True
        )
        result = dq.steadystate(H, Ls, solver=solver)

        assert bool(result.infos.success), (
            f'Solver did not converge (na={na}, nb={nb}, tol={tol:.0e}, '
            f'iters={result.infos.n_iteration})'
        )
        residual = lindbladian_residual(H, Ls, result.rho)
        assert residual < tol, f'Residual {residual:.2e} exceeds tol={tol:.0e}'
        assert_valid_dm(result.rho, precision)


# ---------------------------------------------------------------------------
# Random single-mode tests
# ---------------------------------------------------------------------------


class TestRandomSingleMode:
    """Steady-state solver on random Lindbladians."""

    def test_convergence(self, na, nb, gamma, tol, precision):
        """Solver converges and residual is below tolerance."""
        n = na * nb
        H, Ls = build_random_single_mode(n, seed=0, gamma=gamma)
        solver = SteadyStateGMRES(
            tol=tol, max_iteration=200, krylov_size=32, exact_dm=True
        )
        result = dq.steadystate(H, Ls, solver=solver)

        assert bool(result.infos.success), (
            f'Solver did not converge (n={n}, gamma={gamma}, tol={tol:.0e}, '
            f'iters={result.infos.n_iteration})'
        )
        residual = lindbladian_residual(H, Ls, result.rho)
        assert residual < tol, f'Residual {residual:.2e} exceeds tol={tol:.0e}'
        assert_valid_dm(result.rho, precision)


# ---------------------------------------------------------------------------
# Simple oscillator tests
# ---------------------------------------------------------------------------


class TestSimpleOscillator:
    """Steady-state solver on a driven damped single oscillator."""

    def test_convergence(self, na, tol, precision):
        """Solver converges for H = epsilon_a * a + h.c., L = sqrt(kappa) * a."""
        n = na
        epsilon_a = 0.2
        kappa = 1.0
        a = dq.destroy(n)
        H = epsilon_a * a + epsilon_a * a.dag()
        Ls = [jnp.sqrt(kappa) * a]

        solver = SteadyStateGMRES(
            tol=tol, max_iteration=200, krylov_size=64, exact_dm=True
        )
        result = dq.steadystate(H, Ls, solver=solver)

        assert bool(result.infos.success), (
            f'Solver did not converge (n={n}, tol={tol:.0e}, '
            f'iters={result.infos.n_iteration})'
        )
        residual = lindbladian_residual(H, Ls, result.rho)
        assert residual < tol, f'Residual {residual:.2e} exceeds tol={tol:.0e}'
        assert_valid_dm(result.rho, precision)

        fine_tol = 1e-8 if precision == 'double' else 1e-5
        solver_fine = SteadyStateGMRES(
            tol=fine_tol, max_iteration=200, krylov_size=64, exact_dm=True
        )
        result_fine = dq.steadystate(H, Ls, solver=solver_fine)
        assert bool(result_fine.infos.success), (
            f'Solver did not converge (n={n}, tol={fine_tol:.0e}, '
            f'iters={result_fine.infos.n_iteration})'
        )
        rho_analytical = _simple_oscillator_analytical_steady_state(n, epsilon_a, kappa)
        fidelity = float(dq.fidelity(result_fine.rho, rho_analytical))
        fid_tol = 1e-10 if precision == 'double' else 1e-5
        assert fidelity > 1 - fid_tol, (
            f'Fidelity {fidelity:.8f} too low for analytical steady state '
            f'(target > {1 - fid_tol:.8f})'
        )


# ---------------------------------------------------------------------------
# JIT compatibility tests
# ---------------------------------------------------------------------------


class TestJIT:
    """Verify that steadystate works correctly within a JIT context.

    Since steadystate already applies @jax.jit internally, we test that it can
    be called from within a larger jitted function (composability) and that
    repeated calls hit the compilation cache.
    """

    def test_jit_composability_two_modes(self):
        """steadystate can be called inside a jitted function."""
        na, nb = 12, 3
        H, Ls = build_two_modes(na, nb, kappa_a=1)
        solver = SteadyStateGMRES(tol=1e-1)

        @jax.jit
        def solve_and_get_trace(H, Ls):
            result = dq.steadystate(H, Ls, solver=solver)
            return jnp.trace(result.rho.to_jax()), result.infos.success

        trace_val, success = solve_and_get_trace(H, Ls)
        assert bool(success), 'JIT-wrapped solver did not converge (two-mode)'
        assert jnp.isclose(trace_val, 1.0, atol=1e-4), (
            f'Tr(rho) = {trace_val}, expected 1.0'
        )

        # Second call should use cached compilation
        trace_val2, success2 = solve_and_get_trace(H, Ls)
        assert bool(success2), 'Second JIT call did not converge'

    def test_jit_composability_random_single_mode(self):
        """steadystate can be called inside a jitted function."""
        n = 36
        H, Ls = build_random_single_mode(n, seed=42, gamma=1.0)
        solver = SteadyStateGMRES(tol=1e-1)

        @jax.jit
        def solve_and_get_trace(H, Ls):
            result = dq.steadystate(H, Ls, solver=solver)
            return jnp.trace(result.rho.to_jax()), result.infos.success

        trace_val, success = solve_and_get_trace(H, Ls)
        assert bool(success), 'JIT-wrapped solver did not converge (random)'
        assert jnp.isclose(trace_val, 1.0, atol=1e-4), (
            f'Tr(rho) = {trace_val}, expected 1.0'
        )


# ---------------------------------------------------------------------------
# Double precision sanity check
# ---------------------------------------------------------------------------


def _assert_precision_active(precision):
    x = jnp.array(1.0)
    expected_dtype = jnp.float64 if precision == 'double' else jnp.float32
    assert x.dtype == expected_dtype, f'Expected {expected_dtype} but got {x.dtype}.'


def test_double_precision_active(precision):
    """Backward-compatible name used by older test selections."""
    _assert_precision_active(precision)


def test_precision_active(precision):
    """Current precision sanity check."""
    _assert_precision_active(precision)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
