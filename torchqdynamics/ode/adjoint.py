import torch
import torch.nn as nn
import torchqdynamics.ode as ode

# -------------------------------------------------------------------------------------------------
#     Adjoint integrator base class
# -------------------------------------------------------------------------------------------------

class ODEIntAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, f_adj, y0, tspan, params, solver='dopri5', save_at=(), atol=1e-8, rtol=1e-6):
        # Save into context for backward pass
        ctx.f = f
        ctx.f_adj = f_adj
        ctx.tspan = tspan
        ctx.solver = solver
        ctx.atol = atol
        ctx.rtol = rtol

        # Solve the ODE forward and save results
        with torch.no_grad():
            t, y = ode.odeint(f, y0, tspan, solver, save_at, atol=atol, rtol=rtol)
            ctx.save_for_backward(t, y)
        return t, y

    @staticmethod
    def backward(ctx, *grad_y):
        # Disable gradient computation
        torch.set_grad_enabled(False)
        
        # Unpack context
        f = ctx.f
        f_adj = ctx.f_adj
        tspan = ctx.tspan
        solver = ctx.solver
        atol = ctx.atol
        rtol = ctx.rtol
        t_saved, y_saved = ctx.saved_tensors

        # Get model parameters
        params = torch.cat([p.view(-1) for p in f.parameters() if p.requires_grad])
        params_ = tuple(f.parameters())
        
        # Initialize the augmented state as a flat tensor
        # y is the state, a is the adjoint, g is the gradient w.r.t. parameters
        yT, aT, gT = y_saved[-1], grad_y[-1][-1], torch.zeros_like(params)
        y_len, a_len, g_len = yT.numel(), aT.numel(), gT.numel()
        y_shape, a_shape, g_shape = yT.shape, aT.shape, gT.shape
        aug_y = torch.cat([yT.view(-1), aT.view(-1), gT.view(-1)])

        solver_type = ode.solver_to_solvertype(solver)
        if solver_type == "rk":
            # Define the augmented function for RK solvers
            def aug_f(t, aug_y):
                # Unpack and reshape state
                y, a = aug_y[:y_len], aug_y[y_len:y_len+a_len]
                y, a = y.view(y_shape), a.view(a_shape)
                # Compute the augmented state 
                with torch.enable_grad():
                    y, a = y.requires_grad_(True), a.requires_grad_(True)
                    dy, da = f(t, y), f_adj(t, a)
                    dy, da = dy.view(-1), da.view(-1)
                    # Compute dg/dt = - (d(da/dt)/dtheta @ y)
                    dg = torch.autograd.grad(da, params_, y.view(-1), 
                                            allow_unused=True, retain_graph=True)
                    # Convert back dg to flat tensor
                    dg = torch.cat([torch.zeros_like(param).view(-1) if dgi is None else dgi.view(-1)
                                for param, dgi in zip(params_, dg)]).conj()

                # Repack and return
                return torch.cat([dy, da, dg])
        elif solver_type == "out":
            # Define the augmented function for outsourced solvers
            def aug_f(t, dt, aug_y):
                # Unpack and reshape state
                y, a, g = aug_y[:y_len], aug_y[y_len:y_len+a_len], aug_y[-g_len:]
                y, a, g = y.view(y_shape), a.view(a_shape), g.view(g_shape)
                # Compute the augmented state 
                with torch.enable_grad():
                    y, a = y.requires_grad_(True), a.requires_grad_(True)
                    dy, da = f(t, dt, y), f_adj(t, dt, a)
                    dy, da = dy.view(-1), da.view(-1)
                    # Compute dg/dt = - (d(da/dt)/dtheta @ y)
                    dg = torch.autograd.grad(da, params_, y.view(-1), allow_unused=True)
                    # Convert back dg to flat tensor
                    dg = torch.cat([torch.zeros_like(param).view(-1) if dgi is None else dgi.view(-1)
                                for param, dgi in zip(params_, dg)]).conj()
                    # Add previous value
                    dg = g.view(-1) + dg 

                # Repack and return
                return torch.cat([dy, da, dg])
            
        # Solve the augmented equation backward between every checkpoints
        for i in range(len(t_saved) - 1, 0, -1):
            # Get tspan between checkpoints
            t0, t1 = t_saved[i-1], t_saved[i]
            tspan_ = tspan[torch.logical_and(tspan > t0, tspan < t1)]
            tspan_ = torch.cat([torch.tensor([t0]), tspan_, torch.tensor([t1])])
            # Compute the augmented ODE
            _, aug_y = ode.odeint(aug_f, aug_y, tspan_, solver=solver, save_at=(),
                              atol=atol, rtol=rtol, backward_mode=True)
            # Replace y with its checkpointed version
            aug_y[:y_len] = y_saved[i-1].view(-1)
            # Update adjoint wrt this time point by adding dL / dy(t)
            aug_y[y_len:y_len + a_len] += grad_y[-1][i-1].view(-1)

        a, g = aug_y[y_len:y_len + a_len], aug_y[-g_len:]
        a, g = a.reshape(a_shape), g.reshape(g_shape)
        return (None, None, a, None, g, None, None, None, None)