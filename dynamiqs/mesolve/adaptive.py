from torch import Tensor

from ..solvers.ode.adaptive_solver import AdaptiveSolver
from .me_solver import MESolver


class MEAdaptive(MESolver, AdaptiveSolver):
    def odefun(self, t: float, rho: Tensor) -> Tensor:
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)
        return self.lindbladian(t, rho)
