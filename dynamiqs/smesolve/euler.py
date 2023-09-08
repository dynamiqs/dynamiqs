from torch import Tensor

from ..solvers.ode.fixed_solver import FixedSolver
from .sme_solver import SMESolver


class SMEEuler(SMESolver, FixedSolver):
    def forward(self, t: float, rho: Tensor) -> Tensor:
        # rho: (b_H, b_rho, ntrajs, n, n) -> (b_H, b_rho, ntrajs, n, n)

        # sample Wiener process
        dw = self.sample_wiener(self.dt)

        # update measured signal
        self.update_meas(dw, rho)

        # update state
        drho = self.dt * self.lindbladian(t, rho) + self.diff_backaction(dw, rho)

        return rho + drho
