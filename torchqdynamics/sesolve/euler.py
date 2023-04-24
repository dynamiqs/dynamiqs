from torch import Tensor

from ..ode.ode_forward_solver import ODEForwardSolver


class SEEuler(ODEForwardSolver):
    def __init__(self, *args):
        super().__init__(*args)

        self.H = self.H[:, None, ...]  # (b_H, 1, n, n)

    def forward(self, t: float, psi: Tensor) -> Tensor:
        # Args:
        #     psi: (b_H, b_psi, n, 1)
        #
        # Returns:
        #     (b_H, b_psi, n, 1)

        return psi - self.options.dt * 1j * self.H @ psi
