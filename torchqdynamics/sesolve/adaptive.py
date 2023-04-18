from torch import Tensor

from ..odeint import ForwardQSolver
from ..solver_options import SolverOption
from ..tensor_types import TDOperator


class SEAdaptive(ForwardQSolver):
    def __init__(self, options: SolverOption, H: TDOperator):
        super().__init__(options)

        self.H = H[:, None, ...]  # (b_H, 1, n, n)

    def forward(self, t: float, psi: Tensor) -> Tensor:
        """Compute dpsi / dt = -1j * H(psi) at time t."""
        return -1j * self.H @ psi
