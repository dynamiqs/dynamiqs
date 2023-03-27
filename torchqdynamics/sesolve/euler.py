from torch import Tensor

from ..odeint import ForwardQSolver
from ..solver_options import SolverOption
from ..types import TDOperator


class SEEuler(ForwardQSolver):
    def __init__(self, H: TDOperator, solver_options: SolverOption):
        # Args:
        #     H: (b_H, n, n)

        # convert H to size compatible with (b_H, b_psi, n, n)
        self.H = H[:, None, ...]

        self.options = solver_options
        self.dt = self.options.dt

    def forward(self, t: float, psi: Tensor):
        # Args:
        #     psi: (b_H, b_psi, n, 1)
        #
        # Returns:
        #     (b_H, b_psi, n, 1)

        return psi - self.dt * 1j * self.H @ psi
