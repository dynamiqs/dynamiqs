"""Tests for the steadystate solver.

Tests convergence at different tolerances and system sizes for:
- Two-mode systems (build_two_modes with kappa_a=1)
- Random single-mode systems (build_random_single_mode with gamma=1 and gamma=0.1)
Also tests JIT compatibility and differentiability.
"""

import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.steady_state import SteadyStateGMRES

from .systems import build_random_single_mode, build_two_modes
from .utils import (
    assert_valid_dm,
    lindbladian_residual,
    simple_oscillator_analytical_steady_state,
)

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


@pytest.fixture
def base_tol(precision):
    """Tolerance used for residual checks (not scaled with system size)."""
    return 1e-4 if precision == 'single' else 1e-8


@pytest.fixture
def tol(base_tol, na):
    """Tolerance passed to the solver (scaled with system size)."""
    return base_tol / na


@pytest.fixture(params=[1.0, 0.1], ids=lambda g: f'gamma={g}')
def gamma(request):
    return request.param


# ---------------------------------------------------------------------------
# Two-mode tests
# ---------------------------------------------------------------------------


class TestTwoModes:
    """Steady-state solver on the two-mode cat-qubit system."""

    def test_convergence(self, na, nb, tol, base_tol, precision):
        """Solver converges and residual is below tolerance."""
        H, Ls = build_two_modes(na, nb, kappa_a=1)
        solver = SteadyStateGMRES(
            tol=tol, max_iteration=200, krylov_size=64, exact_dm=True
        )
        result = dq.steadystate(H, Ls, solver=solver)

        residual = lindbladian_residual(H, Ls, result.rho)
        assert residual < base_tol, (
            f'Residual {residual:.2e} exceeds tol={base_tol:.0e}'
        )
        assert_valid_dm(result.rho, precision)


# ---------------------------------------------------------------------------
# Random single-mode tests
# ---------------------------------------------------------------------------


class TestRandomSingleMode:
    """Steady-state solver on random Lindbladians."""

    def test_convergence(self, na, nb, gamma, tol, base_tol, precision):
        """Solver converges and residual is below tolerance."""
        n = na * nb
        H, Ls = build_random_single_mode(n, seed=0, gamma=gamma)
        solver = SteadyStateGMRES(
            tol=tol, max_iteration=200, krylov_size=32, exact_dm=True
        )
        result = dq.steadystate(H, Ls, solver=solver)

        residual = lindbladian_residual(H, Ls, result.rho)
        assert residual < base_tol, (
            f'Residual {residual:.2e} exceeds tol={base_tol:.0e}'
        )
        assert_valid_dm(result.rho, precision)


# ---------------------------------------------------------------------------
# Simple oscillator tests
# ---------------------------------------------------------------------------


class TestSimpleOscillator:
    """Steady-state solver on a driven damped single oscillator."""

    def test_convergence(self, na, tol, base_tol, precision):
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

        residual = lindbladian_residual(H, Ls, result.rho)
        assert residual < base_tol, (
            f'Residual {residual:.2e} exceeds tol={base_tol:.0e}'
        )
        assert_valid_dm(result.rho, precision)

        fine_base_tol = 1e-7 if precision == 'double' else 1e-4
        fine_tol = fine_base_tol / na
        solver_fine = SteadyStateGMRES(
            tol=fine_tol, max_iteration=100, krylov_size=32, exact_dm=True
        )
        result_fine = dq.steadystate(H, Ls, solver=solver_fine)
        rho_analytical = simple_oscillator_analytical_steady_state(n, epsilon_a, kappa)
        fidelity = float(dq.fidelity(result_fine.rho, rho_analytical))
        fid_tol = 1e-8 if precision == 'double' else 1e-5
        assert fidelity > 1 - fid_tol, (
            f'Fidelity {fidelity:.8f} too low for analytical steady state '
            f'(target > {1 - fid_tol:.8f})'
        )


# ---------------------------------------------------------------------------
# JIT compatibility tests
# ---------------------------------------------------------------------------


class TestJIT:
    """Verify that steadystate works correctly within a JIT context."""

    def test_jit_composability_two_modes(self):
        """Steadystate can be called inside a jitted function."""
        na, nb = 12, 3
        H, Ls = build_two_modes(na, nb, kappa_a=1)
        solver = SteadyStateGMRES(tol=1e-1)

        @jax.jit
        def solve_and_get_trace(H, Ls):
            result = dq.steadystate(H, Ls, solver=solver)
            return jnp.trace(result.rho.to_jax())

        trace_val = solve_and_get_trace(H, Ls)
        assert jnp.isclose(trace_val, 1.0, atol=1e-4)

        trace_val2 = solve_and_get_trace(H, Ls)
        assert jnp.isclose(trace_val2, 1.0, atol=1e-4)

    def test_jit_composability_random_single_mode(self):
        """Steadystate can be called inside a jitted function."""
        n = 36
        H, Ls = build_random_single_mode(n, seed=42, gamma=1.0)
        solver = SteadyStateGMRES(tol=1e-1)

        @jax.jit
        def solve_and_get_trace(H, Ls):
            result = dq.steadystate(H, Ls, solver=solver)
            return jnp.trace(result.rho.to_jax())

        trace_val = solve_and_get_trace(H, Ls)
        assert jnp.isclose(trace_val, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Differentiability tests
# ---------------------------------------------------------------------------


class TestAutodiff:
    """Verify that the steady-state pipeline is differentiable."""

    def test_single_mode_jvp(self):
        jax.config.update('jax_enable_x64', True)

        n = 8
        a = dq.destroy(n)
        H0 = 0.5 * a.dag() @ a
        jump_ops = [jnp.sqrt(1.0) * a]
        n_op = a.dag() @ a

        solver = SteadyStateGMRES(
            tol=1e-8, max_iteration=100, krylov_size=32, exact_dm=False
        )

        def loss(epsilon):
            H = H0 + epsilon * (a.dag() + a)
            result = dq.steadystate(H, jump_ops, solver=solver)
            return jnp.real(dq.expect(n_op, result.rho))

        eps0 = jnp.float64(0.8)
        _, grad_jvp = jax.jvp(loss, (eps0,), (jnp.float64(1.0),))

        h = 1e-4
        grad_fd = (loss(eps0 + h) - loss(eps0 - h)) / (2 * h)

        assert jnp.isfinite(grad_jvp)
        assert jnp.abs(grad_jvp) > 1e-8
        assert jnp.isclose(grad_jvp, grad_fd, rtol=5e-2)

    def test_single_mode_grad(self):
        jax.config.update('jax_enable_x64', True)

        n = 8
        a = dq.destroy(n)
        H0 = 0.5 * a.dag() @ a
        jump_ops = [jnp.sqrt(1.0) * a]
        n_op = a.dag() @ a

        solver = SteadyStateGMRES(
            tol=1e-8, max_iteration=100, krylov_size=32, exact_dm=False
        )

        def loss(epsilon):
            H = H0 + epsilon * (a.dag() + a)
            result = dq.steadystate(H, jump_ops, solver=solver)
            return jnp.real(dq.expect(n_op, result.rho))

        eps0 = jnp.float64(0.8)
        grad_rev = jax.grad(loss)(eps0)

        h = 1e-4
        grad_fd = (loss(eps0 + h) - loss(eps0 - h)) / (2 * h)

        assert jnp.isfinite(grad_rev)
        assert jnp.abs(grad_rev) > 1e-8
        assert jnp.isclose(grad_rev, grad_fd, rtol=5e-2)

    def test_two_modes_is_differentiable_wrt_eps_d(self):
        jax.config.update('jax_enable_x64', True)

        na, nb = 12, 5
        solver = SteadyStateGMRES(
            tol=1e-7, max_iteration=200, krylov_size=64, exact_dm=False
        )
        _, b = dq.destroy(na, nb)
        n_b = b.dag() @ b

        def loss(eps_d):
            H, Ls = build_two_modes(na, nb, eps_d=eps_d, kappa_a=1)
            result = dq.steadystate(H, Ls, solver=solver)
            return jnp.real(dq.expect(n_b, result.rho))

        eps0 = jnp.array(2.0)
        _, grad_auto = jax.jvp(loss, (eps0,), (jnp.array(1.0),))

        h = 5e-3
        grad_fd = (loss(eps0 + h) - loss(eps0 - h)) / (2 * h)

        assert jnp.isfinite(grad_auto)
        assert jnp.isfinite(grad_fd)
        assert jnp.abs(grad_auto) > 1e-8
        assert jnp.isclose(grad_auto, grad_fd, rtol=5e-2, atol=1e-4)


# ---------------------------------------------------------------------------
# Precision sanity checks
# ---------------------------------------------------------------------------


def _assert_precision_active(precision):
    x = jnp.array(1.0)
    expected_dtype = jnp.float64 if precision == 'double' else jnp.float32
    assert x.dtype == expected_dtype


def test_double_precision_active(precision):
    _assert_precision_active(precision)


def test_precision_active(precision):
    _assert_precision_active(precision)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
