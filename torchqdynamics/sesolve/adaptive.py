from torch import Tensor

from ..ode_forward_qsolver import ODEForwardQSolver
from ..tensor_types import TDOperator


class SEAdaptive(ODEForwardQSolver):
    def __init__(self, *args, H: TDOperator):
        super().__init__(*args)

        self.H = H[:, None, ...]  # (b_H, 1, n, n)

    def forward(self, t: float, psi: Tensor) -> Tensor:
        """Compute dpsi / dt = -1j * H(psi) at time t."""
        return -1j * self.H @ psi
