from __future__ import annotations

from torch import Tensor

from ..solvers.ode.adaptive_solver import AdaptiveSolver, DormandPrince5


class SEAdaptive(AdaptiveSolver):
    def odefun(self, t: float, psi: Tensor) -> Tensor:
        """Compute dpsi / dt = -1j * H(psi) at time t."""
        # psi: (b_H, b_psi, n, 1) -> (b_H, b_psi, n, 1)
        return -1j * self.H(t) @ psi

    def odefun_adjoint(self, t: float, phi: Tensor) -> Tensor:
        raise NotImplementedError

    def odefun_backward(self, t: float, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def odefun_augmented(
        self, t: float, psi: Tensor, phi: Tensor
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class SEDormandPrince5(SEAdaptive, DormandPrince5):
    pass
