from torch import Tensor

from ..solvers.euler import Euler


class SEEuler(Euler):
    def forward(self, t: float, dt: float, psi: Tensor) -> Tensor:
        # Args:
        #     psi: (b_H, b_psi, n, 1)
        #
        # Returns:
        #     (b_H, b_psi, n, 1)

        return psi - dt * 1j * self.H(t) @ psi
