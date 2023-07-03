from __future__ import annotations

import functools
import warnings
from abc import abstractmethod

import torch
from torch import Tensor
from tqdm.std import TqdmWarning

from ...utils.progress_bar import tqdm
from ...utils.solver_utils import hairer_norm
from ...utils.tensor_types import dtype_complex_to_real
from ..solver import AutogradSolver


class AdaptiveSolver(AutogradSolver):
    def run_autograd(self):
        """Integrate a quantum ODE with an adaptive time step ODE integrator.

        This function integrates an ODE of the form `dy / dt = f(t, y)` with
        `y(0) = y0`. For details about the integration method, see Chapter II.4 of
        `Hairer et al., Solving Ordinary Differential Equations I (1993), Springer
        Series in Computational Mathematics`.
        """
        # initialize the progress bar
        pbar = tqdm(total=self.t_stop[-1].item(), disable=not self.options.verbose)

        # initialize the ODE routine
        t0 = 0.0
        f0 = self.odefun(t0, self.y0)
        dt = self.init_tstep(f0, self.y0, t0)
        error = 1.0
        cache = (dt, error)

        # run the ODE routine
        t, y, ft = t0, self.y0, f0
        step_counter = 0
        for ts in self.t_stop.cpu().numpy():
            while t < ts:
                # update time step
                dt = self.update_tstep(dt, error)

                # check for time overflow
                if t + dt >= ts:
                    cache = (dt, error)
                    dt = ts - t

                # perform a single step of size dt
                ft_new, y_new, y_err = self.step(ft, y, t, dt)

                # compute estimated error of this step
                error = self.get_error(y_err, y, y_new)

                # update if step is accepted
                if error <= 1:
                    t, y, ft = t + dt, y_new, ft_new

                    # update the progress bar
                    pbar.update(dt)

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

    @property
    @abstractmethod
    def order(self) -> int:
        pass

    @property
    @abstractmethod
    def tableau(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def odefun(self, t: float, y: Tensor):
        pass

    @abstractmethod
    def step(
        self, f0: Tensor, y0: Tensor, t0: float, dt: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute a single step of the ODE integration."""
        pass

    @torch.no_grad()
    def get_error(self, y_err: Tensor, y0: Tensor, y1: Tensor) -> float:
        """Compute the error of a given solution.

        See Equation (4.11) of `Hairer et al., Solving Ordinary Differential Equations I
        (1993), Springer Series in Computational Mathematics`.
        """
        scale = self.options.atol + self.options.rtol * torch.max(y0.abs(), y1.abs())
        return hairer_norm(y_err / scale).max().item()

    @torch.no_grad()
    def init_tstep(self, f0: Tensor, y0: Tensor, t0: float) -> float:
        """Initialize the time step of an adaptive step size integrator.

        See Equation (4.14) of `Hairer et al., Solving Ordinary Differential Equations I
        (1993), Springer Series in Computational Mathematics` for the detailed steps.
        For this function, we keep the same notations as in the book.
        """
        sc = self.options.atol + torch.abs(y0) * self.options.rtol
        d0, d1 = hairer_norm(y0 / sc).max().item(), hairer_norm(f0 / sc).max().item()

        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        y1 = y0 + h0 * f0
        f1 = self.odefun(t0 + h0, y1)
        d2 = hairer_norm((f1 - f0) / sc).max().item() / h0
        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1.0 / float(self.order + 1))

        return min(100 * h0, h1)

    @torch.no_grad()
    def update_tstep(self, dt: float, error: float) -> float:
        """Update the time step of an adaptive step size integrator.

        See Equation (4.12) and (4.13) of `Hairer et al., Solving Ordinary Differential
        Equations I (1993), Springer Series in Computational Mathematics` for the
        detailed steps.
        """
        if error == 0:  # no error -> maximally increase the time step
            return dt * self.options.max_factor

        if error <= 1:  # time step accepted -> take next time step at least as large
            return dt * max(
                1.0,
                min(
                    self.options.max_factor,
                    self.options.safety_factor * error ** (-1.0 / self.order),
                ),
            )

        if error > 1:  # time step rejected -> reduce next time step
            return dt * max(
                self.options.min_factor,
                self.options.safety_factor * error ** (-1.0 / self.order),
            )


class DormandPrince5(AdaptiveSolver):
    """Dormand-Prince method for adaptive time step ODE integration.

    This is a fifth order solver that uses a fourth order solution to estimate the
    integration error. It does so using only six function evaluations. See `Dormand and
    Prince, A family of embedded Runge-Kutta formulae (1980), Journal of Computational
    and Applied Mathematics`. See also `Shampine, Some Practical Runge-Kutta Formulas
    (1986), Mathematics of Computation`.
    """

    @functools.cached_property
    def order(self) -> int:
        return 5

    @functools.cached_property
    def tableau(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build the Butcher tableau of the integrator."""
        alpha = [1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0, 0.0]
        beta = [
            [1 / 5, 0, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        ]
        csol5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
        csol4 = [
            5179 / 57600,
            0,
            7571 / 16695,
            393 / 640,
            -92097 / 339200,
            187 / 2100,
            1 / 40,
        ]

        # extract dtype and device from
        dtype = self.y0.dtype
        float_dtype = dtype_complex_to_real(dtype)
        device = self.y0.device

        # initialize tensors
        alpha = torch.tensor(alpha, dtype=float_dtype, device=device)
        beta = torch.tensor(beta, dtype=dtype, device=device)
        csol5 = torch.tensor(csol5, dtype=dtype, device=device)
        csol4 = torch.tensor(csol4, dtype=dtype, device=device)

        return alpha, beta, csol5, csol5 - csol4

    def step(
        self, f: Tensor, y: Tensor, t: float, dt: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        # import butcher tableau
        alpha, beta, csol, cerr = self.tableau

        # compute iterated Runge-Kutta values
        k = torch.empty(7, *f.shape, dtype=f.dtype, device=f.device)
        k[0] = f
        for i in range(1, 7):
            dy = torch.tensordot(dt * beta[i - 1, :i], k[:i], dims=([0], [0]))
            k[i] = self.odefun(t + dt * alpha[i - 1], y + dy)

        # compute results
        f_new = k[-1]
        y_new = y + torch.tensordot(dt * csol[:6], k[:6], dims=([0], [0]))
        y_err = torch.tensordot(dt * cerr, k, dims=([0], [0]))

        return f_new, y_new, y_err
