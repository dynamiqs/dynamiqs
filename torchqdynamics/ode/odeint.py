import warnings
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

from .adjoint import ODEIntAdjoint
from .solver import adapt_step, hairer_norm, init_step, str_to_solver

# --------------------------------------------------------------------------------------
#     Main ODE integrator
# --------------------------------------------------------------------------------------


class Solver(Enum):
    DOPRI5 = 'dopri5'
    RK = 'rk'
    OUT = 'out'


def odeint(
    f,
    y0,
    tspan,
    solver=Solver.DOPRI5,
    save_at=(),
    sensitivity=None,
    model_params=None,
    atol=1e-8,
    rtol=1e-6,
    backward_mode=False,
):
    """Solve an initial value problem determined by function `f` and initial condition
    `y0`.

    If a regular solver is called (such as `solver=Solver.DOPRI5`), this solves the ordinary
    differential equation (ODE) defined by ``dy / dt = f(t, y(t))``. If instead the
    `solver="outsource"` solver is called, this outsources the ODE integration to the
    user. In this case, `f` should be such that ``y(t+dt) = f(t, y(t), dt)``.

    This ODE integrator supports automatic differentiation (AD) either through torch-
    based backward AD, or through a tailor-made adjoint-method AD. The latter is only
    supported for linear ODEs of the form `f(t, y(t)) = A(t) @ y(t)` or `f(t, y(t), dt)
    = B(t, dt) @ y(t)`. For adjoint-method AD, the forward propagation of the adjoint
    state should be implemented in the input model `f`, such that `f.forward_adj(t, a)`
    returns `da/dt` where a is the adjoint state, or `f.forward_adj(t, dt, a)` returns
    `a(t+dt)` in case of the outsourced solver.

    For adjoint-based AD, the model parameters can be supplied either:
        (1) directly in `f`, where `f` is an instance of `nn.Module` with a non-empty
            `f.parameter()` argument;
        (2) through the optional argument `model_params`.

    More details on adjoint-based AD can be found in https://arxiv.org/abs/1806.07366.

    Args:
        f (callable): Function or nn.Module that yields the ODE derivative at time t,
            or the next solution of the ODE at time t for time step dt in case of the
            outsourced solver.
        y0 (tensor): Initial condition of the ODE. tspan (tensor-like): Sorted list,
        array or tensor of time points. For adaptive
            time step solvers, only the first and last values of the tensor are used.
            For fixed time steps solvers, time points correspond to time evaluations.
        solver (str, optional): Solver algorithm. Defaults to an order 5 Dormand-Prince.
            Can also be outsourced by calling `solver="outsource"` or `solver="out"`.
        save_at (tensor-like, optional): Sorted list, array or tensor of time points to
            for which the ODE solution is saved and returned. Defaults to ().
        sensitivity (str, optional): Sensitivity algorithm for AD. Defaults to None.
        model_params(tuple, optional): Tuple of model parameters for adjoint-based AD.
            Defaults to None.
        atol (float, optional): Absolute solver tolerence. Only for adaptive time step.
            Defaults to 1e-8.
        rtol (float, optional): Relative solver tolerence. Only for adaptive time step.
            Defaults to 1e-6.
        backward_mode (bool, optional): Whether to solve the ODE backwards. If true,
            the solution is returned in descending time order. `tspan` should still be
            given in forward-time order. Defaults to False.

    Returns:
        (1-D tensor, N-D tensor): Tuple of tensors containing the saved time points,
            and the ODE solution at these time points.
    """
    tspan = init_tspan(tspan)
    y0, tspan, solver = init_solver(y0, tspan, solver)
    save_at = init_saveat(save_at, tspan, backward_mode)

    # Dispatch to appropriate solver
    if sensitivity in (None, 'autograd'):  # TODO: Separate these two cases
        if solver.solver_type == Solver.RK:
            return odeint_adaptive(
                f, y0, tspan, solver, save_at, atol, rtol, backward_mode
            )
        elif solver.solver_type == Solver.OUT:
            return odeint_outsource(f, y0, tspan, solver, save_at, backward_mode)
    elif sensitivity == 'adjoint':
        return odeint_adjoint(f, y0, tspan, solver, save_at, model_params, atol, rtol)


# --------------------------------------------------------------------------------------
#     ODE subroutines
# --------------------------------------------------------------------------------------


def odeint_adjoint(
    f,
    y0,
    tspan,
    solver=Solver.DOPRI5,
    save_at=(),
    model_params=None,
    atol=1e-8,
    rtol=1e-6,
):
    """Solve an ODE with adjoint method AD in the backward pass."""
    # Extract model parameters
    if model_params is None and not isinstance(f, nn.Module):
        raise TypeError(
            'For adjoint method automatic differentiation, either `f` must be an '
            'instance of `nn.Module`, or parameters should be supplied through the '
            'optional `model_params` argument. Supplying an empty tuple () is also '
            'acceptable if there are no model parameters.'
        )
    if model_params is None:
        n_params = sum(1 for _ in f.parameters())
        params = tuple(p for p in f.parameters() if p.requires_grad)
    else:
        n_params = sum(1 for _ in model_params)
        params = tuple(p for p in model_params if p.requires_grad)
    if len(params) != n_params:
        warnings.warn(
            'Some of the model parameters passed do not require gradients. They will '
            'be excluded from the adjoint-based AD for efficiency.'
        )

    # Warning if using a RK solver with the adjoint method
    if solver_to_solvertype(solver) == Solver.RK:
        warnings.warn(
            'Using a Runga-Kutta solver with adjoint-based automatic differentiation. '
            'Runge-Kutta solvers are numerically unstable in the backward pass and may '
            'diverge. We recommand using a physically-informed solver such as the '
            'Rouchon method, and to use checkpoints in the forward pass (by supplying '
            '`save_at`).'
        )

    # Extract the adjoint state forward method.
    if not (
        hasattr(f.__class__, 'forward_adj') and
        callable(getattr(f.__class__, 'forward_adj'))
    ):
        raise TypeError(
            'For adjoint-based autodiff, `f` must have a method `forward_adj` '
            'with signature `f.forward_adj(t, a)` or `f.forward_adj(t, dt, a)` '
            'in case of the outsourced solver is used.'
        )
    f_adj = f.forward_adj

    # Save at least the first and last elements in the forward pass.
    if len(save_at) == 0 or not save_at[0] == tspan[0]:
        save_at = torch.cat((torch.tensor([tspan[0]]), save_at))
    if len(save_at) == 0 or not save_at[-1] == tspan[-1]:
        save_at = torch.cat((save_at, torch.tensor([tspan[-1]])))

    # Run the ODE integrator
    return ODEIntAdjoint.apply(
        f, f_adj, y0, tspan, solver, save_at, atol, rtol, *params
    )


def odeint_adaptive(
    f_, y0, tspan, solver, save_at=(), atol=1e-8, rtol=1e-6, backward_mode=False
):
    """Core ODE solver subroutine for adaptive step size solvers. Solves an ODE of the
    form `dy / dt = f(t, y(t))`."""
    # Initialize save_at
    if save_at == ():
        save_at = init_saveat(save_at, tspan, backward_mode)

    # Check for backward mode
    if backward_mode:
        f = lambda t, y: -f_(-t, y)
        tspan = -tspan.flip(0)
        save_at = -save_at.flip(0)
    else:
        f = f_

    # Initialize save arrays
    t0, T = tspan[0], tspan[-1]
    t_saved = torch.zeros((len(save_at), ), dtype=t0.dtype)
    y_saved = torch.zeros((len(save_at), ) + y0.shape, dtype=y0.dtype)
    checkpt_counter, checkpt_flag = 0, False
    if save_at[0] == t0:
        t_saved[0], y_saved[0] = t0, y0
        checkpt_counter += 1

    # Initialize the ODE routine
    f0 = f(t0, y0)
    dt = init_step(f, f0, y0, t0, solver.order, atol, rtol)

    # Run the ODE routine
    t, y, ft = t0, y0, f0
    while t < T:
        # Check if checkpointing required, or final time is reached
        if (checkpt_counter < len(save_at)) and (t + dt >= save_at[checkpt_counter]):
            dt_old, checkpt_flag = dt, True
            dt = save_at[checkpt_counter] - t
        elif t + dt >= T:
            dt = T - t

        # Perform a single solver step of size dt
        ft_new, y_new, y_err, stages = solver.step(f, ft, y, t, dt)

        # Compute error
        error_scaled = y_err / (atol + rtol * torch.max(y.abs(), y_new.abs()))
        error_ratio = hairer_norm(error_scaled)
        accept_step = error_ratio <= 1
        # Update results
        if accept_step:
            if checkpt_flag:
                t_saved[checkpt_counter] = t + dt
                y_saved[checkpt_counter] = y_new
                checkpt_counter += 1
                checkpt_flag = False
            t, y = t + dt, y_new
            ft = ft_new
        elif checkpt_flag:
            # reset dt value in case checkpoint flag raised but step not accepted
            dt = dt_old
            checkpt_flag = False
        # Compute new dt
        dt = adapt_step(
            dt,
            error_ratio,
            solver.safety,
            solver.min_factor,
            solver.max_factor,
            solver.order,
        )

    # Return results with forward-valued times
    return -t_saved if backward_mode else t_saved, y_saved


def odeint_outsource(f_, y0, tspan, solver, save_at=(), backward_mode=False):
    """Core ODE solver subroutine for outsourced solvers. Solves an ODE of the form
    `y(t+dt) = f(t, dt, y(t))`."""
    # Initialize `save_at`
    if save_at == ():
        save_at = init_saveat(save_at, tspan, backward_mode)

    # Merge tspan with `save_at` (such that all `save_at` values are in `tspan`)
    tspan = mergesort(tspan, save_at)

    # Check for backward mode
    if backward_mode:
        f = lambda t, dt, y: f_(-t, -dt, y)
        tspan = -tspan.flip(0)
        save_at = -save_at.flip(0)
    else:
        f = f_

    # Initialize save arrays
    t0, T = tspan[0], tspan[-1]
    t_saved = torch.zeros((len(save_at), ), dtype=t0.dtype)
    y_saved = torch.zeros((len(save_at), ) + y0.shape, dtype=y0.dtype)
    checkpt_counter, checkpt_flag = 0, False
    if len(save_at) > 0 and save_at[0] == t0:
        t_saved[0], y_saved[0] = t0, y0
        checkpt_counter += 1

    # Compute time steps
    dtspan = torch.diff(tspan)

    # Run the ODE routine
    y = y0
    for i, t in enumerate(tspan[:-1] + 0.5 * dtspan):
        y = f(t, dtspan[i], y)
        if len(save_at) > 0 and tspan[i + 1] == save_at[checkpt_counter]:
            t_saved[checkpt_counter] = tspan[i + 1]
            y_saved[checkpt_counter] = y
            checkpt_counter += 1

    # Return results with forward-valued times
    return -t_saved if backward_mode else t_saved, y_saved


# --------------------------------------------------------------------------------------
#     Utility functions
# --------------------------------------------------------------------------------------


def init_tspan(tspan):
    """Check tspan is a sorted tensor"""
    if isinstance(tspan, (list, np.ndarray)):
        tspan = torch.cat(tspan)
    if not torch.all(torch.diff(tspan) > 0):
        raise ValueError(
            'Argument `tspan` is not sorted in ascending order '
            'or contains duplicate values.'
        )
    return tspan


def init_solver(y0, tspan, solver):
    """Instantiate the solver and ensure compatibility of dtypes"""
    if isinstance(solver, str):
        solver = str_to_solver(solver, y0.dtype)
    y0, tspan = solver.sync_device_dtype(y0, tspan)
    return y0, tspan, solver


def init_saveat(save_at, tspan, backward_mode=False):
    """Prepare `save_at` tensor by trimming values below t0 and above T"""
    # Check type
    if isinstance(save_at, tuple):
        save_at = torch.tensor(save_at)
    elif isinstance(save_at, (list, np.ndarray)):
        save_at = torch.cat(save_at)

    # Check if sorted
    if not torch.all(torch.diff(save_at) > 0):
        raise ValueError(
            'Argument `save_at` is not sorted or contains duplicate values.'
        )

    # Always save last time point (or first in backward mode)
    t0, T = tspan[0], tspan[-1]
    save_at = save_at[torch.logical_and(save_at >= t0, save_at <= T)]
    if not backward_mode:
        if len(save_at) == 0 or not save_at[-1] == T:
            save_at = torch.cat((save_at, torch.tensor([T])))
    else:
        if len(save_at) == 0 or not save_at[0] == t0:
            save_at = torch.cat((torch.tensor([t0]), save_at))

    return save_at


def count_params(model):
    """Counter the total number of parameters with required gradient in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mergesort(a, b):
    """Merge two sorted tensors into a sorted tensor while removing duplicates."""
    c = torch.cat([a, b])
    c, _ = torch.sort(c)
    flag = torch.ones(c.numel(), dtype=bool)
    torch.ne(c[1:], c[:-1], out=flag[1:])
    return c[flag]
