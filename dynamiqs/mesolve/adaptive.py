from torch import Tensor

from ..solvers.ode.adaptive_solver import AdaptiveSolver, DormandPrince5
from .me_solver import MESolver


class MEAdaptive(MESolver, AdaptiveSolver):
    def odefun(self, t: float, rho: Tensor) -> Tensor:
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)
        return self.lindbladian(t, rho)


class MEDormandPrince5(MEAdaptive, DormandPrince5):
    pass
