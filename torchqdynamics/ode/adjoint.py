import torch
import torch.nn as nn

import torchqdynamics.ode as ode

# -------------------------------------------------------------------------------------------------
#     Adjoint integrator base class
# -------------------------------------------------------------------------------------------------


class ODEIntAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, f_adj, y0, tspan, solver, save_at, atol, rtol, *params):
        # Save into context for backward pass
        ctx.f = f
        ctx.f_adj = f_adj
        ctx.tspan = tspan
        ctx.solver = solver
        ctx.atol = atol
        ctx.rtol = rtol

        # Solve the ODE forward, save results and save model parameters
        with torch.no_grad():
            t, y = ode.odeint(f, y0, tspan, solver, save_at, atol=atol, rtol=rtol)
            ctx.save_for_backward(t, y, *params)
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
        t_saved, y_saved, *params = ctx.saved_tensors

        # Get number of model parameters, and the initial shape of params before flattening
        n_params = sum(p.numel() for p in params)
        params_shapes = tuple(p.shape for p in params)
        params_numels = torch.tensor([p.numel() for p in params])
        params_indices = torch.cat(
            [torch.tensor([0]), torch.cumsum(params_numels, dim=0)]
        )

        # Initialize the augmented state as a flat tensor
        # y is the state, a is the adjoint, g is the gradient w.r.t. parameters
        yT, aT, gT = y_saved[-1], grad_y[-1][-1], torch.zeros(
            n_params, dtype=y_saved.dtype
        )
        y_len, a_len, g_len = yT.numel(), aT.numel(), gT.numel()
        y_shape, a_shape, g_shape = yT.shape, aT.shape, gT.shape
        aug_y = torch.cat([yT.view(-1), aT.view(-1), gT.view(-1)])

        solver_type = grt.solver_to_solvertype(solver)
        if solver_type == "rk":
            # Define the augmented function for RK solvers
            def aug_f(t, aug_y):
                # Unpack and reshape state
                y, a = aug_y[:y_len], aug_y[y_len:y_len + a_len]
                y, a = y.view(y_shape), a.view(a_shape)
                # Compute the augmented state
                with torch.enable_grad():
                    y, a = y.requires_grad_(True), a.requires_grad_(True)
                    dy, da = f(t, y), f_adj(t, a)
                    dy, da = dy.view(-1), da.view(-1)
                    # Compute dg/dt = - (d(da/dt)/dtheta @ y)
                    dg = torch.autograd.grad(
                        da, params, y.view(-1), allow_unused=True, retain_graph=True
                    )
                    # Convert back dg to flat tensor
                    dg = torch.cat(
                        [
                            torch.zeros_like(p).view(-1)
                            if dg_ is None else dg_.view(-1)
                            for p, dg_ in zip(params, dg)
                        ]
                    )

                # Repack and return
                return torch.cat([dy, da, dg])

        elif solver_type == "out":
            # Define the augmented function for outsourced solvers
            def aug_f(t, dt, aug_y):
                # Unpack and reshape state
                y, a, g = aug_y[:y_len], aug_y[y_len:y_len + a_len], aug_y[-g_len:]
                y, a, g = y.view(y_shape), a.view(a_shape), g.view(g_shape)

                # Compute the augmented state
                with torch.enable_grad():
                    y, a = y.requires_grad_(True), a.requires_grad_(True)
                    dy, da = f(t, dt, y), f_adj(t, dt, a)
                    dy, da = dy.view(-1), da.view(-1)
                    # Compute dg = - (d(da)/dtheta @ y)
                    dg = torch.autograd.grad(
                        da, params, y.view(-1), allow_unused=True, retain_graph=True
                    )
                    # Convert back dg to flat tensor
                    dg = torch.cat(
                        [
                            torch.zeros_like(p).view(-1)
                            if dg_ is None else dg_.view(-1)
                            for p, dg_ in zip(params, dg)
                        ]
                    )
                    # Add previous value
                    dg = g.view(-1) + dg

                # Repack and return
                return torch.cat([dy, da, dg])

        # Solve the augmented equation backward between every checkpoints
        for i in range(len(t_saved) - 1, 0, -1):
            # Get tspan between checkpoints
            t0, t1 = t_saved[i - 1], t_saved[i]
            tspan_ = tspan[torch.logical_and(tspan > t0, tspan < t1)]
            tspan_ = torch.cat([torch.tensor([t0]), tspan_, torch.tensor([t1])])
            # Compute the augmented ODE
            _, aug_y = ode.odeint(
                aug_f, aug_y, tspan_, solver=solver, save_at=(), atol=atol, rtol=rtol,
                backward_mode=True
            )
            aug_y = aug_y[-1]
            # Replace y with its checkpointed version
            aug_y[:y_len] = y_saved[i - 1].view(-1)
            # Update adjoint wrt this time point by adding dL / dy(t)
            aug_y[y_len:y_len + a_len] += grad_y[-1][i - 1].view(-1)

        # Extract the relevant gradients to return
        a, g = aug_y[y_len:y_len + a_len], aug_y[-g_len:]

        # Reshape outputs
        adjoint_out = a.reshape(a_shape)
        grads_out = tuple(
            g[params_indices[i]:params_indices[i + 1]].reshape(params_shapes[i])
            for i in range(len(params))
        )

        # Convert gradients of real-valued parameters to real-valued gradients
        grads_out = tuple(
            g.real if p.is_floating_point() else g for (g, p) in zip(grads_out, params)
        )

        return (None, None, adjoint_out, None, None, None, None, None, *grads_out)
