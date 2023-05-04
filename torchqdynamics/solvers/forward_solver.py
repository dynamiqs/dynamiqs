from abc import abstractmethod

import torch

from ..utils.progress_bar import tqdm
from .solver import Solver


class ForwardSolver(Solver):
    GRADIENT_ALG = ['autograd']

    def integrate_nograd(self):
        with torch.no_grad():
            self.integrate_autograd()

    def integrate_autograd(self):
        """Integrate a quantum ODE with a fixed time step custom integrator.

        Note:
            The solver times are defined using `torch.linspace` which ensures that the
            overall solution is evolved from the user-defined time (up to an error of
            `rtol=1e-5`). However, this may induce a small mismatch between the time
            step inside `solver` and the time step inside the iteration loop. A small
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
            y = self.forward(t, dt, y)

        # save final time step (`t` goes `0.0` to `t_save[-1]` excluded)
        self.save(y)

    def integrate_adjoint(self):
        return NotImplementedError(
            'This solver does not support adjoint-based gradient computation.'
        )

    @abstractmethod
    def forward(self, t, y, dt):
        pass
