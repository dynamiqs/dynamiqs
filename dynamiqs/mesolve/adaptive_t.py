from __future__ import annotations

import torch
from torch import Tensor

from ..solvers.ode.adaptive_solver import AdjointAdaptiveSolver, DormandPrince5
from ..solvers.solver import Solver
from ..solvers.utils import cache


class METSolver(Solver):
    def __init__(self, *args, L: Tensor):
        super().__init__(*args)
        self.L = L  # (nL, ..., n, n)

        # define cached sum of LdagL
        self.sum_LdagL = cache(lambda L: (L.mH @ L).sum(dim=0))  # (..., n, n)

        # define identity operator
        n = self.H.size(-1)
        self.I = torch.eye(n, device=self.device, dtype=self.cdtype)  # (n, n)

        # define cached non-hermitian Hamiltonian
        self.Hnh = cache(lambda H, sum_LdagL: H - 0.5j * sum_LdagL)  # (..., n, n)

    def lindbladian(self, t: float, rho: Tensor) -> Tensor:
        """Compute the action of the Lindbladian on the density matrix.

        Notes:
            Hermiticity of the output is enforced to avoid numerical instability
            with Runge-Kutta solvers.
        """
        # rho: (..., n, n) -> (..., n, n)
        H = self.H(t)
        L = self.L(t)
        sum_LdagL = self.sum_LdagL(L)
        Hnh = self.Hnh(H, sum_LdagL)
        out = -1j * Hnh @ rho + 0.5 * (L @ rho @ L.mH).sum(0)
        return out + out.mH

    def adjoint_lindbladian(self, t: float, phi: Tensor) -> Tensor:
        """Compute the action of the adjoint Lindbladian on an operator.

        Notes:
            Hermiticity of the output is enforced to avoid numerical instability
            with Runge-Kutta solvers.
        """
        # phi: (..., n, n) -> (..., n, n)
        H = self.H(t)
        L = self.L(t)
        sum_LdagL = self.sum_LdagL(L)
        Hnh = self.Hnh(H, sum_LdagL)
        out = 1j * Hnh.mH @ phi + 0.5 * (L.mH @ phi @ L).sum(0)
        return out + out.mH


class METAdaptive(METSolver, AdjointAdaptiveSolver):
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


class METDormandPrince5(METAdaptive, DormandPrince5):
    pass
