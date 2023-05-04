from torch import Tensor

from ..solvers.ode.forward_solver import ForwardSolver


class SEEuler(ForwardSolver):
    def forward(self, t: float, dt: float, psi: Tensor) -> Tensor:
        # Args:
        #     psi: (b_H, b_psi, n, 1)
        #
        # Returns:
        #     (b_H, b_psi, n, 1)

        return psi - dt * 1j * self.H(t) @ psi
