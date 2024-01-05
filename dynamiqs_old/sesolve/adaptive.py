from __future__ import annotations

from torch import Tensor

from ..solvers.ode.adaptive_solver import AdjointAdaptiveSolver, DormandPrince5


class SEAdaptive(AdjointAdaptiveSolver):
    def odefun(self, t: float, psi: Tensor) -> Tensor:
        """Compute dpsi / dt = -1j * H(psi) at time t."""
        # psi: (..., n, 1) -> (..., n, 1)
        return -1j * self.H(t) @ psi

    def odefun_backward(self, t: float, psi: Tensor) -> Tensor:
        # psi: (..., n, 1) -> (..., n, 1)
        # t is passed in as negative time
        raise NotImplementedError

    def odefun_adjoint(self, t: float, phi: Tensor) -> Tensor:
        # phi: (..., n, 1) -> (..., n, 1)
        # t is passed in as negative time
        raise NotImplementedError


class SEDormandPrince5(SEAdaptive, DormandPrince5):
    pass
