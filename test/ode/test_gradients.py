from test.problems import PROBLEMS, PROBLEMS_OUT

import pytest
import torch
import torch.nn as nn

from torchqdynamics.ode.odeint import odeint, odeint_adjoint

SOLVERS = ['dopri5']
SOLVERS_OUT = ['out']
DTYPES = [torch.float64]
TEST_ITERATIONS = range(1)

# -------------------------------------------------------------------------------------------------
#     Problem tests for regular solvers
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("prob", PROBLEMS.keys())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("test_iter", TEST_ITERATIONS)
def test_gradcheck_odeint(prob, solver, dtype, test_iter):
    """Test odeint for multiple test problems"""
    p = PROBLEMS[prob](dtype)
    func = lambda y0: odeint(p, y0, p.tspan, save_at=p.tspan, solver=solver, atol=p.
                             atol, rtol=p.rtol)
    assert torch.autograd.gradcheck(func, (p.y0), eps=1e-2, atol=1e-3)


@pytest.mark.parametrize("prob", ['linear'])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("test_iter", TEST_ITERATIONS)
def test_gradcheck_odeint_adjoint(prob, solver, dtype, test_iter):
    """Test odeint_adjoint for multiple test problems"""
    p = PROBLEMS[prob](dtype)

    def func(cst, y0):
        t, y = odeint_adjoint(p, y0, p.tspan, save_at=p.tspan, solver=solver,
                              atol=p.atol, rtol=p.rtol)
        return torch.linalg.norm(y[-1])**2

    assert torch.autograd.gradcheck(func, (p.cst, p.y0), eps=1e-4, atol=1e-3)


# -------------------------------------------------------------------------------------------------
#     Problems tests for outsourced solvers
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("prob", PROBLEMS_OUT.keys())
@pytest.mark.parametrize("solver", SOLVERS_OUT)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("test_iter", TEST_ITERATIONS)
def test_gradcheck_odeint_out(prob, solver, dtype, test_iter):
    """Test outsourced odeint for multiple test problems"""
    p = PROBLEMS_OUT[prob](dtype)
    func = lambda y0: odeint(p, y0, p.tspan, save_at=p.save_at, solver=solver, atol=p.
                             atol, rtol=p.rtol)
    assert torch.autograd.gradcheck(func, (p.y0), eps=1e-2, atol=1e-3)


@pytest.mark.parametrize("prob", PROBLEMS_OUT.keys())
@pytest.mark.parametrize("solver", SOLVERS_OUT)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("test_iter", TEST_ITERATIONS)
def test_gradcheck_odeint_adjoint_out(prob, solver, dtype, test_iter):
    """Test outsourced odeint_adjoint for multiple test problems"""
    p = PROBLEMS_OUT[prob](dtype)

    def func(cst, y0):
        t, y = odeint_adjoint(p, y0, p.tspan, save_at=p.save_at, solver=solver,
                              atol=p.atol, rtol=p.rtol)
        return torch.linalg.norm(y[-1])**2

    assert torch.autograd.gradcheck(func, (p.cst, p.y0), eps=1e-6, atol=1e-3)