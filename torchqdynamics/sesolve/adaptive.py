from torch import Tensor

from ..odeint import ForwardQSolver
from ..solver_options import SolverOption
from ..types import TDOperator


class SEAdaptive(ForwardQSolver):
    def __init__(self, H: TDOperator, solver_options: SolverOption):
        self.H = H[:, None, ...]  # (b_H, 1, n, n)
        self.options = solver_options

    def forward(self, t: float, psi: Tensor) -> Tensor:
        """Compute dpsi / dt = -1j * H(psi) at time t."""
        return -1j * self.H @ psi
