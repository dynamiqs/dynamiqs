from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor

from .adaptive import DormandPrince45
from .progress_bar import tqdm
from .qsolver import QSolver
from .solver_options import Dopri45, ODEAdaptiveStep, ODEFixedStep


class ODEForwardQSolver(QSolver):
    @abstractmethod
    def forward(self, t: float, y: Tensor) -> Tensor:
        """Iterate the quantum state forward.

        Args:
            t: Time.
            y: Quantum state, of shape `(..., m, n)`.

        Returns:
            Tensor of shape `(..., m, n)`.
        """
        pass

    def run(self):
        """Integrate a quantum ODE starting from an initial state.

        The ODE is solved from time `t=0.0` to `t=t_save[-1]`.
        """
        # dispatch to appropriate odeint subroutine
        if self.gradient_alg is None:
            self._odeint_inplace()
        elif self.gradient_alg == 'autograd':
            self._odeint_main()

    def _odeint_inplace(self):
        """Integrate a quantum ODE with an in-place solver.

        Simple solution for now so torch does not store gradients.
        TODO Implement a genuine in-place solver.
        """
        with torch.no_grad():
            self._odeint_main()

    def _odeint_main(self):
        """Dispatch the ODE integration to fixed or adaptive time step subroutines."""
        if isinstance(self.options, ODEFixedStep):
            self._fixed_odeint()
        elif isinstance(self.options, ODEAdaptiveStep):
            self._adaptive_odeint()

    def _fixed_odeint(self):
        """Integrate a quantum ODE with a fixed time step solver.

        Note:
            The solver times are defined using `torch.linspace` which ensures that the
            overall solution is evolved from the user-defined time (up to an error of
            `rtol=1e-5`). However, this may induce a small mismatch between the time
            step inside `qsolver` and the time step inside the iteration loop. A small
            error can thus buildup throughout the ODE integration. TODO Fix this.
        """
        # get time step
        dt = self.options.dt

        # assert that `t_save` values are multiples of `dt`
        if not torch.allclose(torch.round(self.t_save / dt), self.t_save / dt):
            raise ValueError(
                'For fixed time step solvers, every value of `t_save` must be a'
                ' multiple of the time step `dt`.'
            )

        # define time values
        num_times = torch.round(self.t_save[-1] / dt).int() + 1
        times = torch.linspace(0.0, self.t_save[-1], num_times)

        # run the ode routine
        y = self.y0
        for t in tqdm(times[:-1], disable=not self.options.verbose):
            # save solution
            if t >= self.next_tsave():
                self.save(y)

            # iterate solution
            y = self.forward(t, y)

        # save final time step (`t` goes `0.0` to `t_save[-1]` excluded)
        self.save_final(y)

    def _adaptive_odeint(self):
        """Integrate a quantum ODE with an adaptive time step solver.

        This function integrates an ODE of the form `dy / dt = f(t, y)` with
        `y(0) = y0`, using a Runge-Kutta adaptive time step solver.

        For details about the integration method, see Chapter II.4 of `Hairer et al.,
        Solving Ordinary Differential Equations I (1993), Springer Series in
        Computational Mathematics`.
        """
        # save initial solution
        self.save(self.y0)

        save_flag = False

        # initialize the adaptive solver
        args = (
            self.forward,
            self.options.factor,
            self.options.min_factor,
            self.options.max_factor,
            self.options.atol,
            self.options.rtol,
        )
        if isinstance(self.options, Dopri45):
            solver = DormandPrince45(*args)

        # initialize the ODE routine
        t0 = 0.0
        f0 = solver.f(t0, self.y0)
        dt = solver.init_tstep(f0, self.y0, t0)

        # initialize the progress bar
        pbar = tqdm(total=self.t_save[-1].item(), disable=not self.options.verbose)

        # run the ODE routine
        t, y, ft = t0, self.y0, f0
        step_counter, max_steps = 0, self.options.max_steps
        next_tsave = self.next_tsave()
        while t < self.t_save[-1] and step_counter < max_steps:
            # if a time in `t_save` is reached, raise a flag and rescale dt accordingly
            if t + dt >= next_tsave:
                save_flag = True
                dt_old = dt
                dt = next_tsave - t

            # perform a single solver step of size dt
            ft_new, y_new, y_err = solver.step(ft, y, t, dt)

            # compute estimated error of this step
            error = solver.get_error(y_err, y, y_new)

            # update results if step is accepted
            if error <= 1:
                t, y, ft = t + dt, y_new, ft_new

                # update the progress bar
                pbar.update(dt.item())

                # save results if flag is raised
                if save_flag:
                    self.save(y)

            # return to the original dt, lower the flag and get next save time
            if save_flag:
                dt = dt_old
                save_flag = False
                next_tsave = self.next_tsave()

            # compute the next dt
            dt = solver.update_tstep(dt, error)
            step_counter += 1

        # save last state if not already done
        self.save_final(y)
