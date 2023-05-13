import warnings
from abc import abstractmethod

import torch
from torch import Tensor
from tqdm.std import TqdmWarning

from ...options import Dopri5
from ...utils.progress_bar import tqdm
from ..solver import AutogradSolver
from .adaptive_integrator import DormandPrince5


class AdaptiveSolver(AutogradSolver):
    def run_autograd(self):
        """Integrate a quantum ODE with an adaptive time step ODE integrator.

        This function integrates an ODE of the form `dy / dt = f(t, y)` with
        `y(0) = y0`, using a Runge-Kutta adaptive time step integrator.

        For details about the integration method, see Chapter II.4 of `Hairer et al.,
        Solving Ordinary Differential Equations I (1993), Springer Series in
        Computational Mathematics`.
        """
        # initialize the adaptive integrator
        args = (
            self.odefun,
            self.options.factor,
            self.options.min_factor,
            self.options.max_factor,
            self.options.atol,
            self.options.rtol,
        )
        if isinstance(self.options, Dopri5):
            integrator = DormandPrince5(*args)

        # initialize the progress bar
        pbar = tqdm(total=self.t_save[-1].item(), disable=not self.options.verbose)

        # initialize the ODE routine
        t0 = torch.tensor(0.0)
        f0 = integrator.f(t0, self.y0)
        dt = integrator.init_tstep(f0, self.y0, t0)
        error = torch.tensor(1.0)
        cache = (dt, error)

        # run the ODE routine
        t, y, ft = t0, self.y0, f0
        step_counter = 0
        for ts in self.t_save:
            while t < ts:
                # update time step
                dt = integrator.update_tstep(dt, error)

                # check for time overflow
                if t + dt >= ts:
                    cache = (dt, error)
                    dt = ts - t

                # perform a single ODE integrator step of size dt
                ft_new, y_new, y_err = integrator.step(ft, y, t, dt)

                # compute estimated error of this step
                error = integrator.get_error(y_err, y, y_new)

                # update if step is accepted
                if error <= 1:
                    t, y, ft = t + dt, y_new, ft_new

                    # update the progress bar
                    pbar.update(dt.item())

                # raise error if max_steps reached
                step_counter += 1
                if step_counter == self.options.max_steps:
                    raise RuntimeError(
                        'Maximum number of time steps reached. Consider using lower'
                        ' order integration methods, or raising the maximum number of'
                        ' time steps with the `options` argument.'
                    )

            # save solution
            self.save(y)

            # use cache to retrieve time step and error
            dt, error = cache

        # close progress bar
        with warnings.catch_warnings():  # ignore tqdm precision overflow
            warnings.simplefilter('ignore', TqdmWarning)
            pbar.close()

    @abstractmethod
    def odefun(self, t: float, y: Tensor):
        pass
