import pytest

import torch
import torch.nn as nn
from torchqdynamics.ode.odeint import odeint, odeint_adjoint
from test.problems import PROBLEMS

SOLVERS = ['dopri5']
DTYPES = [torch.float64]
TEST_ITERATIONS = range(1)

@pytest.mark.parametrize("prob", PROBLEMS.keys())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("test_iter", TEST_ITERATIONS)
def test_gradcheck_odeint(prob, solver, dtype, test_iter):
    """Test odeint for multiple test problems"""
    p = PROBLEMS[prob](dtype)
    func = lambda y0: odeint(p, y0, p.tspan, save_at=p.tspan, solver=solver, 
                                    atol=p.atol, rtol=p.rtol)
    assert torch.autograd.gradcheck(func, (p.y0), eps=1e-2, atol=1e-3)

@pytest.mark.parametrize("prob", ['linear'])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("test_iter", TEST_ITERATIONS)
def test_gradcheck_odeint_adjoint(prob, solver, dtype, test_iter):
    """Test odeint for multiple test problems"""
    p = PROBLEMS[prob](dtype)
    def func(cst, y0):
        t, y = odeint_adjoint(p, y0, p.tspan, save_at=p.tspan, solver=solver, 
                              atol=p.atol, rtol=p.rtol)
        return torch.linalg.norm(y[-1])**2
    assert torch.autograd.gradcheck(func, (p.cst, p.y0), eps=1e-4, atol=1e-3)

