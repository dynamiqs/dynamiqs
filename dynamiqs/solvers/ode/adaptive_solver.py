from __future__ import annotations

import functools
import warnings
from abc import abstractmethod

import torch
from torch import Tensor
from tqdm.std import TqdmWarning

from ..solver import AdjointSolver, AutogradSolver
from ..utils.utils import add_tuples, hairer_norm, none_to_zeros_like, tqdm
from .adjoint_autograd import AdjointAutograd


class AdaptiveSolver(AutogradSolver):
    """Integrate an ODE of the form $dy / dt = f(y, t)$ in forward time with initial
    condition $y(t_0)$ using an adaptive step-size integrator.

    For details about the integration method, see Chapter II.4 of [1].

    [1] Hairer et al., Solving Ordinary Differential Equations I (1993), Springer
        Series in Computational Mathematics.
    """

    @property
    @abstractmethod
    def order(self) -> int:
        pass

    @property
    @abstractmethod
    def tableau(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def odefun(self, t: float, y: Tensor) -> Tensor:
        """Returns $f(y, t)$."""
        pass

    @abstractmethod
    def step(
        self, t0: float, y0: Tensor, f0: Tensor, dt: float, fun: callable
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute a single step of the ODE integration."""
        pass

    def run_autograd(self):
        """Integrates the ODE forward from time `self.t0` to time `self.tstop[-1]`
        starting from initial state `self.y0`, and save the state for each time in
        `self.tstop`."""

        # initialize the progress bar
        self.pbar = tqdm(total=self.tstop[-1].item(), disable=not self.options.verbose)

        # initialize the step counter
        self.step_counter = 0

        # initialize the ODE routine
        f0 = self.odefun(self.t0, self.y0)
        dt = self.init_tstep(self.t0, self.y0, f0, self.odefun)
        error = 1.0

        # run the ODE routine
        t, y, ft = self.t0, self.y0, f0
        for tnext in self.tstop.cpu().numpy():
            y, ft, dt, error = self.integrate(t, tnext, y, ft, dt, error)
            self.save(y)
            t = tnext

        # close the progress bar
        with warnings.catch_warnings():  # ignore tqdm precision overflow
            warnings.simplefilter('ignore', TqdmWarning)
            self.pbar.close()

    def integrate(
        self, t0: float, t1: float, y: Tensor, ft: Tensor, dt: float, error: float
    ) -> tuple[Tensor, Tensor, float, float]:
        """Integrates the ODE forward from time `t0` to time `t1` with initial state
        `y`."""
        cache = (dt, error)
        t = t0
        while t < t1:
            dt = self.update_tstep(dt, error)

            # check for time overflow
            if t + dt >= t1:
                cache = (dt, error)
                dt = t1 - t

            # compute the next step
            ft_new, y_new, y_err = self.step(t, y, ft, dt, self.odefun)
            error = self.get_error(y_err, y, y_new)
            if error <= 1:
                t, y, ft = t + dt, y_new, ft_new
                self.pbar.update(dt)

            # check max steps are not reached
            self.increment_step_counter(t)

        dt, error = cache
        return y, ft, dt, error

    def increment_step_counter(self, t: float):
        """Increment the step counter and check for max steps."""
        self.step_counter += 1
        if self.step_counter == self.options.max_steps:
            raise RuntimeError(
                'Maximum number of time steps reached in adaptive time step ODE'
                f' solver at time t={t:.2g} (`max_steps={self.options.max_steps}`).'
                ' This is likely due to a diverging solution. Try increasing the'
                ' maximum number of steps, or use a different solver.'
            )

    @torch.no_grad()
    def get_error(self, y_err: Tensor, y0: Tensor, y1: Tensor) -> float:
        """Compute the error of a given solution.

        See Equation (4.11) of [1].
        """
        scale = self.options.atol + self.options.rtol * torch.max(y0.abs(), y1.abs())
        return hairer_norm(y_err / scale).max().item()

    @torch.no_grad()
    def init_tstep(self, t0: float, y0: Tensor, f0: Tensor, fun: callable) -> float:
        """Initialize the time step of an adaptive step size integrator.

        See Equation (4.14) of [1] for the detailed steps. For this function, we keep
        the same notations as in the book.
        """
        sc = self.options.atol + torch.abs(y0) * self.options.rtol
        d0, d1 = hairer_norm(y0 / sc).max().item(), hairer_norm(f0 / sc).max().item()

        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        y1 = y0 + h0 * f0
        f1 = fun(t0 + h0, y1)
        d2 = hairer_norm((f1 - f0) / sc).max().item() / h0
        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1.0 / float(self.order + 1))

        return min(100 * h0, h1)

    @torch.no_grad()
    def update_tstep(self, dt: float, error: float) -> float:
        """Update the time step of an adaptive step size integrator.

        See Equation (4.12) and (4.13) of [1] for the detailed steps.
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


class AdjointAdaptiveSolver(AdaptiveSolver, AdjointSolver):
    """Integrate an augmented ODE of the form $(1) dy / dt = fy(y, t),
    $(2) da / dt = fa(a, y)$ in backward time with initial condition $y(t_0)$ using an
    adaptive step-size integrator."""

    @abstractmethod
    def odefun_backward(self, t: float, y: Tensor) -> Tensor:
        """Returns $fy(y, t)$."""
        pass

    @abstractmethod
    def odefun_adjoint(self, t: float, a: Tensor) -> Tensor:
        """Returns $fa(a, t)$."""
        pass

    def odefun_augmented(self, t: float, y: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        """Returns $fy(y, t)$ and $fa(a, t)$."""
        return self.odefun_backward(t, y), self.odefun_adjoint(t, a)

    def run_adjoint(self):
        AdjointAutograd.apply(self, self.y0, *self.options.params)

    def init_augmented(
        self, t0: float, y: Tensor, a: Tensor
    ) -> tuple[Tensor, Tensor, float, float]:
        f0, l0 = self.odefun_augmented(t0, y, a)
        dt_y = self.init_tstep(-t0, y, f0, self.odefun_backward)
        dt_a = self.init_tstep(-t0, a, l0, self.odefun_adjoint)
        dt = min(dt_y, dt_a)
        error = 1.0
        return f0, l0, dt, error

    def integrate_augmented(
        self,
        t0: float,
        t1: float,
        y: Tensor,
        a: Tensor,
        g: tuple[Tensor, ...],
        ft: Tensor,
        lt: Tensor,
        dt: float,
        error: float,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, ...], Tensor, Tensor, float, float]:
        """Integrates the augmented ODE forward from time `t0` to `t1` (with
        `t0` < `t1` < 0) starting from initial state `(y, a)`."""
        cache = (dt, error)

        t = t0
        while t < t1:
            # update time step
            dt = self.update_tstep(dt, error)

            # check for time overflow
            if t + dt >= t1:
                cache = (dt, error)
                dt = t1 - t

            with torch.enable_grad():
                # perform a single step of size dt
                ft_new, y_new, y_err = self.step(-t, y, ft, dt, self.odefun_backward)
                lt_new, a_new, a_err = self.step(-t, a, lt, dt, self.odefun_adjoint)

                # compute estimated error of this step
                error_a = self.get_error(a_err, a, a_new)
                error_y = self.get_error(y_err, y, y_new)
                error = max(error_a, error_y)

                # update if step is accepted
                if error <= 1:
                    t, y, a, ft, lt = t + dt, y_new, a_new, ft_new, lt_new

                    # compute g(t-dt)
                    dg = torch.autograd.grad(
                        a,
                        self.options.params,
                        y,
                        allow_unused=True,
                        retain_graph=True,
                    )
                    dg = none_to_zeros_like(dg, self.options.params)
                    g = add_tuples(g, dg)

                    # update the progress bar
                    self.pbar.update(dt)

            # free the graph of y and a
            y, a, ft, lt = y.data, a.data, ft.data, lt.data

        dt, error = cache
        return y, a, g, ft, lt, dt, error


class DormandPrince5(AdjointAdaptiveSolver):
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

        # initialize tensors
        alpha = torch.tensor(alpha, dtype=self.rdtype, device=self.device)
        beta = torch.tensor(beta, dtype=self.cdtype, device=self.device)
        csol5 = torch.tensor(csol5, dtype=self.cdtype, device=self.device)
        csol4 = torch.tensor(csol4, dtype=self.cdtype, device=self.device)

        return alpha, beta, csol5, csol5 - csol4

    def step(
        self, t: float, y: Tensor, f: Tensor, dt: float, fun: callable
    ) -> tuple[Tensor, Tensor, Tensor]:
        # import butcher tableau
        alpha, beta, csol, cerr = self.tableau

        # compute iterated Runge-Kutta values
        k = torch.empty(7, *f.shape, dtype=self.cdtype, device=self.device)
        k[0] = f
        for i in range(1, 7):
            dy = torch.tensordot(dt * beta[i - 1, :i], k[:i], dims=([0], [0]))
            k[i] = fun(t + dt * alpha[i - 1].item(), y + dy)

        # compute results
        f_new = k[-1]
        y_new = y + torch.tensordot(dt * csol[:6], k[:6], dims=([0], [0]))
        y_err = torch.tensordot(dt * cerr, k, dims=([0], [0]))

        return f_new, y_new, y_err
