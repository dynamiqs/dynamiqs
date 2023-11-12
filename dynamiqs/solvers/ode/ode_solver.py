import warnings
from abc import abstractmethod
from typing import Any

from torch import Tensor
from tqdm import TqdmWarning

from ..solver import AdjointSolver, AutogradSolver
from ..utils.utils import tqdm
from .adjoint_autograd import AdjointAutograd


class ODESolver(AutogradSolver):
    """Integrate an ODE of the form $dy / dt = f(y, t)$ in forward time with initial
    condition $y(t_0)$ using an ODE integrator."""

    def __init__(self, *args):
        super().__init__(*args)

        # initialize the progress bar
        self.pbar = tqdm(total=self.tstop[-1], disable=not self.options.verbose)

    def init_forward(self) -> tuple:
        # initial values of the ODE routine
        return self.t0, self.y0

    def run_autograd(self):
        self.run_forward()

    def run_forward(self):
        """Integrates the ODE forward from time `self.t0` to time `self.tstop[-1]`
        starting from initial state `self.y0`, and save the state for each time in
        `self.tstop`."""

        # initialize the ODE routine
        t, y, *args = self.init_forward()

        # run the ODE routine
        for tnext in self.tstop:
            y, *args = self.integrate(t, tnext, y, *args)
            self.save(y)
            t = tnext

        # close the progress bar
        with warnings.catch_warnings():  # ignore tqdm precision overflow
            warnings.simplefilter('ignore', TqdmWarning)
            self.pbar.close()

    @abstractmethod
    def integrate(self, t0: float, t1: float, y: Tensor, *args: Any) -> tuple:
        """Integrates the ODE forward from time `t0` to time `t1` with initial state
        `y`."""
        pass


class AdjointODESolver(ODESolver, AdjointSolver):
    """Integrate an augmented ODE of the form $(1) dy / dt = fy(y, t)$ and
    $(2) da / dt = fa(a, y)$ in backward time with initial condition $y(t_0)$ using an
    ODE integrator."""

    def run_adjoint(self):
        AdjointAutograd.apply(self, self.y0, *self.options.params)

    def init_augmented(self, t0: float, y0: Tensor, a0: Tensor) -> tuple:
        return t0, y0, a0

    @abstractmethod
    def integrate_augmented(
        self,
        t0: float,
        t1: float,
        y: Tensor,
        a: Tensor,
        g: tuple[Tensor, ...],
        *args: Any,
    ) -> tuple:
        """Integrates the augmented ODE forward from time `t0` to `t1` (with
        `t0` < `t1` < 0) starting from initial state `(y, a)`."""
        pass
