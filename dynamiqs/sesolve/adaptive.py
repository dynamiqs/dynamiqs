from torch import Tensor

from ..solvers.ode.adaptive_solver import AdaptiveSolver, DormandPrince5


class SEAdaptive(AdaptiveSolver):
    def odefun(self, t: float, psi: Tensor) -> Tensor:
        """Compute dpsi / dt = -1j * H(psi) at time t."""
        # psi: (b_H, b_psi, n, 1) -> (b_H, b_psi, n, 1)
        return -1j * self.H(t) @ psi


class SEDormandPrince5(SEAdaptive, DormandPrince5):
    pass
