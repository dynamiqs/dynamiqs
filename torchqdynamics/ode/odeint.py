import torch
import torch.nn as nn
import numpy as np
from .solver import str_to_solver, init_step, adapt_step, hairer_norm
from .adjoint import ODEIntAdjoint

# -------------------------------------------------------------------------------------------------
#     Main ODE integrator
# -------------------------------------------------------------------------------------------------

def odeint(f, y0, tspan, solver='dopri5', save_at=(), sensitivity=None,
           atol=1e-8, rtol=1e-6, backward_mode=False):
    """Solve an initial value problem determined by function `f` and initial condition `y0`.

    If a Runge-Kutta solver is called (such as 'dopri5'), this solves the ordinary 
    differential equation (ODE) defined by ``dy / dt = f(t, y(t))``. If instead a Rouchon
    solver is called, this solves the ODE defined by ``y(t+dt) = f(t, y(t), dt)``.

    Automatic differentiation (AD) is supported either through torch-based backward AD,
    or through a custom-made adjoint state AD. The latter is only supported for linear 
    functions of the form `f(t, y(t)) = f(t) y(t)` or `f(t, y(t), dt) = f(t, dt) y(t)`.
    In this case, the forward propagation of the adjoint state should be implemented in
    the input model `f`, according to `def forward_adj(self, t, a) -> da/dt` which returns
    the forward-pass derivative of the adjoint state `a` at time `t`.

    Args:
        f (callable): Function or nn.Module that yields the ODE derivative at time t,
            or the next solution of the ODE at time t for time step dt in case of a
            Rouchon solver. 
        y0 (tensor): Initial condition of the ODE.
        tspan (tensor-like): Sorted list, array or tensor of time points. For adaptive 
            solvers, only the first and last values of the tensor are used.
        solver (str, optional): Solver algorithm. Defaults to an order 5 Dormand-Prince.
        save_at (tensor-like, optional): Sorted list, array or tensor of time points to
            for which the ODE solution is saved and returned. Defaults to ().
        sensitivity (str, optional): Sensitivity algorithm for AD. Defaults to None.
        atol (float, optional): Absolute solver tolerence. Only for adaptive time step.
            Defaults to 1e-8.
        rtol (float, optional): Relative solver tolerence. Only for adaptive time step.
            Defaults to 1e-6.
        backward_mode (bool, optional): Whether to solve the ODE backwards. If true,
            the solution is returned in descending time order. `tspan` should still be
            given in forward-time order. Defaults to False.

    Returns:
        (tensor, tensor): Tuple of tensors containing the saved time points, and the ODE
            solution at these time points.
    """
    tspan = init_tspan(tspan)
    y0, tspan, solver = init_solver(y0, tspan, solver)
    save_at = init_saveat(save_at, tspan)

    # Send to appropriate solver
    if sensitivity in (None, "autograd"): #TODO: Separate these two cases
        if solver.solver_type == "rk":
            return odeint_adaptive(f, y0, tspan, solver, save_at, atol, rtol, backward_mode)
        elif solver.solver_type == "rouchon":
            return odeint_rouchon(f, y0, tspan, solver, save_at, atol, rtol, backward_mode)
    elif sensitivity == "adjoint":
        return odeint_adjoint(f, y0, tspan, solver, save_at, atol, rtol)


# -------------------------------------------------------------------------------------------------
#     ODE subroutines
# -------------------------------------------------------------------------------------------------

def odeint_adjoint(f, y0, tspan, solver='dopri5', save_at=(), atol=1e-8, rtol=1e-6):
    """Solve an ODE with an adjoint method in the backward pass."""
    # Check inputs
    if not isinstance(f, nn.Module) or count_params(f) == 0:
        raise TypeError("For adjoint-based autodiff, `f` must be an instance of nn.Module "
                        "with a non-empty set of parameters.")

    if not (hasattr(f.__class__, 'forward_adj') and callable(getattr(f.__class__, 'forward_adj'))):
        raise TypeError("For adjoint-based autodiff, `f` must have a method "
                        "`forward_adj(self, t, a)`.")

    # Extract model parameters
    params = torch.cat([p.view(-1) for p in f.parameters() if p.requires_grad])
    f_adj = f.forward_adj
    # Run the ODE integrator
    t, y = ODEIntAdjoint.apply(f, f_adj, y0, tspan, params, solver, save_at, atol, rtol)
    return t, y

def odeint_adaptive(f_, y0, tspan, solver, save_at=(), atol=1e-8, rtol=1e-6, backward_mode=False):
    """Core ODE solver subroutine for adaptive step solvers."""
    # Check for backward mode
    if backward_mode:
        f = lambda t, y: -f_(-t, y)
        tspan = -tspan.flip(0)
        save_at = -save_at.flip(0)
    else: f = f_

    # Initialize save arrays
    t0, T = tspan[0], tspan[-1]
    t_saved = torch.zeros((len(save_at),), dtype=t0.dtype)
    y_saved = torch.zeros((len(save_at),) + y0.shape, dtype=y0.dtype)
    ckpt_counter, ckpt_flag = 0, False
    if len(save_at) > 0 and save_at[0] == t0:
        t_saved[0], y_saved[0] = t0, y0
        ckpt_counter += 1

    # Initialize the ODE routine
    f0 = f(t0, y0)
    dt = init_step(f, f0, y0, t0, solver.order, atol, rtol)
    
    # Run the ODE routine
    t, y, ft = t0, y0, f0
    while t < T:         
        # Check if checkpointing required, or final time is reached
        if len(save_at) > 0 and (ckpt_counter < len(save_at)) and (t + dt >= save_at[ckpt_counter]):
            dt_old, ckpt_flag = dt, True
            dt = save_at[ckpt_counter] - t
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
            if ckpt_flag:
                t_saved[ckpt_counter] = t + dt
                y_saved[ckpt_counter] = y_new
                ckpt_counter += 1
                ckpt_flag = False
            t, y = t + dt, y_new
            ft = ft_new
        elif ckpt_flag:
            # reset dt value in case checkpoint flag raised but step not accepted
            dt = dt_old
            ckpt_flag = False
        # Compute new dt
        dt = adapt_step(dt, error_ratio, solver.safety, solver.min_factor, 
                        solver.max_factor, solver.order)

    # Return results with forward-valued times
    if backward_mode:
        t, t_saved = -t, -t_saved
    if len(save_at) == 0:
        return t, y
    else:
        return t_saved, y_saved

def odeint_rouchon(f, y0, tspan, solver, save_at=(), atol=1e-8, rtol=1e-6, backward_mode=False):
    return
# -------------------------------------------------------------------------------------------------
#     Utility functions
# -------------------------------------------------------------------------------------------------

def init_tspan(tspan):
    """Check tspan is a sorted tensor"""
    if isinstance(tspan, (list, np.ndarray)): 
        tspan = torch.cat(tspan)
    if not torch.all(torch.diff(tspan) > 0):
        raise ValueError("Argument `tspan` is not sorted in ascending order "
                         "or contains duplicate values.")
    return tspan

def init_solver(y0, tspan, solver):
    """Instantiate the solver and ensure compatibility of dtypes"""
    if isinstance(solver, str):
        solver = str_to_solver(solver, y0.dtype)
    y0, tspan = solver.sync_device_dtype(y0, tspan)
    return y0, tspan, solver

def init_saveat(save_at, tspan):
    """Prepare `save_at` tensor by trimming values below t0 and above T"""
    # Check type
    if isinstance(save_at, tuple):
        save_at = torch.tensor(save_at)
    elif isinstance(save_at, (list, np.ndarray)): 
        save_at = torch.cat(save_at)
    # Check if sorted
    t0, T = tspan[0], tspan[-1]
    if not torch.all(torch.diff(save_at) > 0):
        raise ValueError("Argument `save_at` is not sorted or contains duplicate values.")
    save_at = save_at[torch.logical_and(save_at >= t0, save_at < T)]
    # Always save last time point. 
    if len(save_at) > 0:
        save_at = torch.cat((save_at, torch.tensor([T])))
    return save_at

def count_params(model):
    """Counter the total number of parameters with required gradient in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)