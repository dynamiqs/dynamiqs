from torch import Tensor

from ..solvers.ode.adaptive_solver import AdaptiveSolver


class SEAdaptive(AdaptiveSolver):
    def odefun(self, t: float, psi: Tensor) -> Tensor:
        """Compute dpsi / dt = -1j * H(psi) at time t."""
        return -1j * self.H(t) @ psi
