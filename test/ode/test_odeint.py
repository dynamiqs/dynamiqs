from test.problems import PROBLEMS, PROBLEMS_OUT

import pytest
import torch
import torch.nn as nn
from torchdyn.numerics.odeint import odeint as odeint_tdyn

from torchqdynamics.ode.odeint import odeint, odeint_adjoint

SOLVERS = ['dopri5']
SOLVERS_OUT = ['out']
DTYPES = [torch.float32, torch.float64]
TEST_ITERATIONS = range(1)

# -------------------------------------------------------------------------------------------------
#     Problem tests for regular solvers
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("prob", PROBLEMS.keys())
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("test_iter", TEST_ITERATIONS)
def test_odeint(prob, solver, dtype, test_iter):
    """Test odeint for multiple test problems"""
    p = PROBLEMS[prob](dtype)
    # Compute solutions
    yt_exp = p.solution()
    t, yt = odeint(p, p.y0, p.tspan, save_at=p.tspan, solver=solver, atol=p.atol,
                   rtol=p.rtol)
    _, yt_tdyn = odeint_tdyn(p, p.y0, p.tspan, solver, atol=p.atol, rtol=p.rtol)
    # Check solutions
    assert torch.norm((yt - yt_exp) / yt_exp) < p.test_tol
    assert torch.norm((yt - yt_tdyn) / yt_tdyn) < p.test_tol
    # Check output dtype
    assert yt.dtype == p.dtype
    assert t.dtype == dtype
    # Check save times
    assert torch.allclose(t, p.tspan.to(dtype))


@pytest.mark.parametrize("prob", PROBLEMS.keys())
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("test_iter", TEST_ITERATIONS)
def test_odeint_adjoint(prob, solver, dtype, test_iter):
    """Test odeint_adjoint for multiple test problems. Only the forward pass is tested."""
    p = PROBLEMS[prob](dtype)
    # Compute solutions
    yt_exp = p.solution()
    t, yt = odeint_adjoint(p, p.y0, p.tspan, save_at=p.tspan, solver=solver,
                           atol=p.atol, rtol=p.rtol)
    _, yt_tdyn = odeint_tdyn(p, p.y0, p.tspan, solver, atol=p.atol, rtol=p.rtol)
    # Check solutions
    assert torch.norm((yt - yt_exp) / yt_exp) < p.test_tol
    assert torch.norm((yt - yt_tdyn) / yt_tdyn) < p.test_tol
    # Check output dtype
    assert yt.dtype == p.dtype
    assert t.dtype == dtype
    # Check save times
    assert torch.allclose(t, p.tspan.to(dtype))


@pytest.mark.parametrize("prob", PROBLEMS.keys())
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("test_iter", TEST_ITERATIONS)
def test_odeint_backward(prob, solver, dtype, test_iter):
    """Test odeint for multiple test problems"""
    p = PROBLEMS[prob](dtype)
    # Compute solutions
    yt_exp = p.solution()
    t, yt = odeint(p, p.y0, p.tspan, save_at=p.tspan, solver=solver, atol=p.atol,
                   rtol=p.rtol, backward_mode=False)
    t_bw, yt_bw = odeint(p, yt[-1], p.tspan, save_at=p.tspan, solver=solver,
                         atol=p.atol, rtol=p.rtol, backward_mode=True)
    # Check solutions
    assert torch.norm((yt - yt_bw.flip(0)) / yt) < p.test_tol


# -------------------------------------------------------------------------------------------------
#     Problems tests for outsourced solvers
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("prob", PROBLEMS_OUT.keys())
@pytest.mark.parametrize("solver", SOLVERS_OUT)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("test_iter", TEST_ITERATIONS)
def test_odeint_out(prob, solver, dtype, test_iter):
    """Test outsourced odeint for multiple test problems"""
    p = PROBLEMS_OUT[prob](dtype)
    p_ref = PROBLEMS[prob](dtype)
    # Compute solutions
    yt_exp = p.solution()
    t, yt = odeint(p, p.y0, p.tspan, save_at=p.save_at, solver=solver, atol=p.atol,
                   rtol=p.rtol)
    _, yt_tdyn = odeint_tdyn(p_ref, p_ref.y0, p_ref.tspan, 'dopri5', atol=p_ref.atol,
                             rtol=p_ref.rtol)
    # Check solutions
    assert torch.norm((yt - yt_exp) / yt_exp) < p.test_tol
    assert torch.norm((yt - yt_tdyn) / yt_tdyn) < p.test_tol
    # Check output dtype
    assert yt.dtype == p.dtype
    assert t.dtype == dtype
    # Check save times
    assert torch.allclose(t, p.save_at.to(dtype))


@pytest.mark.parametrize("prob", PROBLEMS_OUT.keys())
@pytest.mark.parametrize("solver", SOLVERS_OUT)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("test_iter", TEST_ITERATIONS)
def test_odeint_out_adjoint(prob, solver, dtype, test_iter):
    """Test outsourced odeint_adjoint for multiple test problems. Only the forward pass is tested."""
    p = PROBLEMS_OUT[prob](dtype)
    p_ref = PROBLEMS[prob](dtype)
    # Compute solutions
    yt_exp = p.solution()
    t, yt = odeint_adjoint(p, p.y0, p.tspan, save_at=p.save_at, solver=solver,
                           atol=p.atol, rtol=p.rtol)
    _, yt_tdyn = odeint_tdyn(p_ref, p_ref.y0, p_ref.tspan, 'dopri5', atol=p_ref.atol,
                             rtol=p_ref.rtol)
    # Check solutions
    assert torch.norm((yt - yt_exp) / yt_exp) < p.test_tol
    assert torch.norm((yt - yt_tdyn) / yt_tdyn) < p.test_tol
    # Check output dtype
    assert yt.dtype == p.dtype
    assert t.dtype == dtype
    # Check save times
    assert torch.allclose(t, p.save_at.to(dtype))


@pytest.mark.parametrize("prob", PROBLEMS_OUT.keys())
@pytest.mark.parametrize("solver", SOLVERS_OUT)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("test_iter", TEST_ITERATIONS)
def test_odeint_out_backward(prob, solver, dtype, test_iter):
    """Test odeint for multiple test problems"""
    p = PROBLEMS_OUT[prob](dtype)
    # Compute solutions
    yt_exp = p.solution()
    t, yt = odeint(p, p.y0, p.tspan, save_at=p.save_at, solver=solver, atol=p.atol,
                   rtol=p.rtol, backward_mode=False)
    t_bw, yt_bw = odeint(p, yt[-1], p.tspan, save_at=p.save_at, solver=solver,
                         atol=p.atol, rtol=p.rtol, backward_mode=True)
    # Check solutions
    assert torch.norm((yt - yt_bw.flip(0)) / yt) < p.test_tol
