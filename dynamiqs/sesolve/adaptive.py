from __future__ import annotations

from torch import Tensor

from ..solvers.ode.adaptive_solver import AdaptiveSolver, DormandPrince5


class SEAdaptive(AdaptiveSolver):
    def odefun(self, t: float, psi: Tensor) -> Tensor:
        """Compute dpsi / dt = -1j * H(psi) at time t."""
        # psi: (b_H, 1, b_psi, n, 1) -> (b_H, 1, b_psi, n, 1)
        return -1j * self.H(t) @ psi

    def odefun_backward(self, t: float, psi: Tensor) -> Tensor:
        # psi: (b_H, 1, b_psi, n, 1) -> (b_H, 1, b_psi, n, 1)
        # t is passed in as negative time
        raise NotImplementedError

    def odefun_adjoint(self, t: float, phi: Tensor) -> Tensor:
        # phi: (b_H, 1, b_psi, n, 1) -> (b_H, 1, b_psi, n, 1)
        # t is passed in as negative time
        raise NotImplementedError


class SEDormandPrince5(SEAdaptive, DormandPrince5):
    pass
