import torch

from ..odeint import ForwardQSolver
from ..solver_options import SolverOption
from ..types import TimeDependentOperator


class SEEuler(ForwardQSolver):
    def __init__(self, H: TimeDependentOperator, solver_options: SolverOption):
        # Args:
        #     H: (b_H, n, n)

        # convert H to size compatible with (b_H, b_psi, n, n)
        self.H = H[:, None, ...]

        self.options = solver_options

    def forward(self, t: float, dt: float, psi: torch.Tensor):
        # Args:
        #     psi: (b_H, b_psi, n, 1)
        #
        # Returns:
        #     (b_H, b_psi, n, 1)

        return psi - dt * 1j * self.H @ psi
