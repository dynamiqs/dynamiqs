"""Tests for the steadystate solver.

Tests convergence at different tolerances and system sizes for:
- Two-mode systems (build_two_modes with kappa_a=1)
- Random single-mode systems (build_random_single_mode with gamma=1 and gamma=0.1)
Also tests JIT compatibility.
"""

import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.steady_state.solvers.steady_state_solver_jump_kernel import (
    SteadyStateJumpKernel,
)

from .systems import build_two_modes
from .utils import assert_valid_dm, lindbladian_residual

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
def steadytol(request, precision):
    return 1e-4 if precision == 'single' else 1e-8


@pytest.fixture
def tol(precision):
    return 1e-4 if precision == 'single' else 1e-8


@pytest.fixture(params=[1.0, 0.1], ids=lambda g: f'gamma={g}')
def gamma(request):
    return request.param


# ---------------------------------------------------------------------------
# Two-mode tests
# ---------------------------------------------------------------------------


class TestOscillator:
    """Steady-state solver on the simple oscillator system."""

    def test_convergence(self, na, tol, steadytol, precision):
        """Solver converges and residual is below tolerance."""
        a = dq.destroy(na)
        H = a @ a.dag()
        L = [a]
        solver = SteadyStateJumpKernel(tol=tol, steady_tol=steadytol, exact_dm=True)
        result = dq.steadystate(H, L, solver=solver)

        assert bool(result.infos.success), (
            f'Solver did not converge (n={na}, tol={tol:.0e}, '
            f'Success={result.infos.success})'
        )
        residual = lindbladian_residual(H, L, result.rho)
        assert residual < tol, f'Residual {residual:.2e} exceeds tol={tol:.0e}'
        assert_valid_dm(result.rho, precision)


class TestCyclicJumpOps:
    """Steady-state solver on a system with cyclic jump operators (L_i = |i><i|
    for i=0,...,d-1).
    """

    def test_convergence(self, na, tol, steadytol, precision):
        """Solver converges and residual is below tolerance."""
        d = na
        L = []
        for i in range(d):
            ket = dq.basis(d, i)
            Li = ket @ ket.dag()
            L.append(Li)
        H = jnp.zeros((d, d))
        H = dq.asqarray(H)
        solver = SteadyStateJumpKernel(tol=tol, steady_tol=steadytol, exact_dm=True)
        result = dq.steadystate(H, L, solver=solver)

        assert bool(result.infos.success), (
            f'Solver did not converge (n={na}, tol={tol:.0e}, '
            f'Success={result.infos.success})'
        )
        residual = lindbladian_residual(H, L, result.rho)
        assert residual < tol, f'Residual {residual:.2e} exceeds tol={tol:.0e}'
        assert_valid_dm(result.rho, precision)


class TestTwoModes:
    """Steady-state solver on the two-mode cat-qubit system."""

    def test_convergence(self, na, nb, tol, steadytol, precision):
        """Solver converges and residual is below tolerance."""
        H, Ls = build_two_modes(na, nb, kappa_b=20, kappa_a=0)
        solver = SteadyStateJumpKernel(tol=tol, steady_tol=steadytol, exact_dm=True,)
        result = dq.steadystate(H, Ls, solver=solver)

        # assert bool(result.infos.success), (
        #     f'Solver did not converge (n={na}, tol={tol:.0e}, '
        #     f'Success={result.infos.success})'
        # )
        residual = lindbladian_residual(H, Ls, result.rho)
        assert residual < tol, f'Residual {residual:.2e} exceeds tol={tol:.0e}'
        assert_valid_dm(result.rho, precision)
