from torch import Tensor

from ..odeint import ForwardQSolver
from ..solver_options import SolverOption
from ..tensor_types import TDOperator


class SEEuler(ForwardQSolver):
    def __init__(self, options: SolverOption, H: TDOperator):
        # Args:
        #     H: (b_H, n, n)
        super().__init__(options)

        # convert H to size compatible with (b_H, b_psi, n, n)
        self.H = H[:, None, ...]

    def forward(self, t: float, psi: Tensor):
        # Args:
        #     psi: (b_H, b_psi, n, 1)
        #
        # Returns:
        #     (b_H, b_psi, n, 1)

        return psi - self.options.dt * 1j * self.H @ psi
