from torch import Tensor

from ..ode.forward_solver import ForwardSolver


class SEAdaptive(ForwardSolver):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, t: float, psi: Tensor) -> Tensor:
        """Compute dpsi / dt = -1j * H(psi) at time t."""
        return -1j * self.H(t) @ psi
