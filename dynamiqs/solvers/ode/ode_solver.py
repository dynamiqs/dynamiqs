import warnings
from abc import abstractmethod
from typing import Any

from torch import Tensor
from tqdm import TqdmWarning

from ..solver import AutogradSolver
from ..utils.utils import tqdm


class ODESolver(AutogradSolver):
    """Integrate an ODE of the form $dy / dt = f(y, t)$ in forward time with initial
    condition $y(t_0)$ using an ODE integrator."""

    def __init__(self, *args):
        super().__init__(*args)

        # initialize the progress bar
        self.pbar = tqdm(total=float(self.tstop[-1]), disable=not self.options.verbose)

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
