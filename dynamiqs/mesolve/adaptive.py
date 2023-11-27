from __future__ import annotations

from torch import Tensor

from ..solvers.ode.adaptive_solver import AdjointAdaptiveSolver, DormandPrince5
from .me_solver import MESolver


class MEAdaptive(MESolver, AdjointAdaptiveSolver):
    def odefun(self, t: float, rho: Tensor) -> Tensor:
        # rho: (..., n, n) -> (..., n, n)
        return self.lindbladian(t, rho)

    def odefun_backward(self, t: float, rho: Tensor) -> Tensor:
        # rho: (..., n, n) -> (..., n, n)
        # t is passed in as negative time
        return -self.lindbladian(-t, rho)

    def odefun_adjoint(self, t: float, phi: Tensor) -> Tensor:
        # phi: (..., n, n) -> (..., n, n)
        # t is passed in as negative time
        return self.adjoint_lindbladian(-t, phi)


class MEDormandPrince5(MEAdaptive, DormandPrince5):
    pass
