from torch import Tensor

from ..solvers.ode.fixed_solver import FixedSolver


class SEEuler(FixedSolver):
    def forward(self, t: float, dt: float, psi: Tensor) -> Tensor:
        # psi: (b_H, b_psi, n, 1) -> (b_H, b_psi, n, 1)
        return psi - dt * 1j * self.H(t) @ psi
