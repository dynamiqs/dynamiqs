from torch import Tensor

from ..solvers.ode.fixed_solver import FixedSolver


class SEEuler(FixedSolver):
    def forward(self, t: float, psi: Tensor) -> Tensor:
        # psi: (..., n, 1) -> (..., n, 1)
        return psi - self.dt * 1j * self.H(t) @ psi
