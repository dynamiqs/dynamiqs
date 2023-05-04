from torch import Tensor

from ..ode.forward_solver import ForwardSolver


class SEEuler(ForwardSolver):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, t: float, psi: Tensor) -> Tensor:
        # Args:
        #     psi: (b_H, b_psi, n, 1)
        #
        # Returns:
        #     (b_H, b_psi, n, 1)

        return psi - self.options.dt * 1j * self.H(t) @ psi
