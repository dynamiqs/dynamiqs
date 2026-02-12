"""Tests for the steady_state solver.

Tests convergence at different tolerances and system sizes for:
- Two-mode systems (build_two_modes with kappa_a=1)
- Random single-mode systems (build_random_single_mode with gamma=1 and gamma=0.1)
Also tests JIT compatibility.
"""

import jax

import jax.numpy as jnp
import dynamiqs as dq
import pytest

from dynamiqs.steady_state.api.steady_state_solver import steady_state

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
    # Ls_q = dq.stack(Ls)
    L_rho = dq.lindbladian(H, Ls, rho)
    return float(jnp.max(jnp.abs(L_rho.to_jax())))


def _dm_atol(precision):
    """Return the absolute tolerance for density matrix checks."""
    return 1e-8 if precision == 'double' else 1e-4


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
        rho, info = steady_state(
            H, Ls, tol=tol, max_iter=200, krylov_size=64, exact_dm=False
        )

        assert bool(info.success), (
            f'Solver did not converge (na={na}, nb={nb}, tol={tol:.0e}, '
            f'iters={info.n_iteration})'
        )
        residual = lindbladian_residual(H, Ls, rho)
        assert residual < tol, f'Residual {residual:.2e} exceeds tol={tol:.0e}'
        assert_valid_dm(rho, precision)


# ---------------------------------------------------------------------------
# Random single-mode tests
# ---------------------------------------------------------------------------


class TestRandomSingleMode:
    """Steady-state solver on random Lindbladians."""

    def test_convergence(self, na, nb, gamma, tol, precision):
        """Solver converges and residual is below tolerance."""
        n = na * nb
        H, Ls = build_random_single_mode(n, seed=0, gamma=gamma)
        rho, info = steady_state(H, Ls, tol=tol, max_iter=200, krylov_size=32)

        assert bool(info.success), (
            f'Solver did not converge (n={n}, gamma={gamma}, tol={tol:.0e}, '
            f'iters={info.n_iteration})'
        )
        residual = lindbladian_residual(H, Ls, rho)
        assert residual < tol, f'Residual {residual:.2e} exceeds tol={tol:.0e}'
        assert_valid_dm(rho, precision)


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

        rho, info = steady_state(
            H, Ls, tol=tol, max_iter=200, krylov_size=64, exact_dm=True
        )

        assert bool(info.success), (
            f'Solver did not converge (n={n}, tol={tol:.0e}, iters={info.n_iteration})'
        )
        residual = lindbladian_residual(H, Ls, rho)
        assert residual < tol, f'Residual {residual:.2e} exceeds tol={tol:.0e}'
        assert_valid_dm(rho, precision)


# ---------------------------------------------------------------------------
# JIT compatibility tests
# ---------------------------------------------------------------------------


class TestJIT:
    """Verify that steady_state is jittable."""

    def test_jit_two_modes(self):
        na, nb = 12, 3
        H, Ls = build_two_modes(na, nb, kappa_a=1)
        tol = 1e-1

        @jax.jit
        def solve():
            return steady_state(H, Ls, tol=tol)

        # First call triggers compilation
        rho, info = solve()
        assert bool(info.success), 'JIT solver did not converge (two-mode)'
        assert lindbladian_residual(H, Ls, rho) < tol

        # Second call uses cached compilation
        rho2, info2 = solve()
        assert bool(info2.success), 'Second JIT call did not converge'

    def test_jit_random_single_mode(self):
        n = 36
        H, Ls = build_random_single_mode(n, seed=42, gamma=1.0)
        tol = 1e-1

        @jax.jit
        def solve():
            return steady_state(H, Ls, tol=tol)

        rho, info = solve()
        assert bool(info.success), 'JIT solver did not converge (random)'
        assert lindbladian_residual(H, Ls, rho) < tol


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
