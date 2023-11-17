import torch
from torch import Tensor

from ..solvers.solver import Solver
from ..solvers.utils import cache, kraus_map


class MESolver(Solver):
    def __init__(self, *args, jump_ops: Tensor):
        super().__init__(*args)
        self.L = jump_ops  # (1, b_L, 1, len(L), n, n)
        self.sum_LdagL = torch.sum(self.L.mH @ self.L, dim=0)  # (1, b_L, 1, n, n)

        # define identity operator
        n = self.H.size(-1)
        self.I = torch.eye(n, device=self.device, dtype=self.cdtype)  # (n, n)

        # define cached non-hermitian Hamiltonian
        self.Hnh = cache(lambda H: H - 0.5j * self.sum_LdagL)  # (b_H, b_L, 1, n, n)

    def lindbladian(self, t: float, rho: Tensor) -> Tensor:
        """Compute the action of the Lindbladian on the density matrix.

        Notes:
            Hermiticity of the output is enforced to avoid numerical instability
            with Runge-Kutta solvers.
        """
        H = self.H(t)
        out = -1j * self.Hnh(H) @ rho + 0.5 * kraus_map(rho, self.L)
        return out + out.mH  # (b_H, b_L, b_rho, n, n)

    def adjoint_lindbladian(self, t: float, phi: Tensor) -> Tensor:
        """Compute the action of the adjoint Lindbladian on an operator.

        Notes:
            Hermiticity of the output is enforced to avoid numerical instability
            with Runge-Kutta solvers.
        """
        H = self.H(t)
        out = 1j * self.Hnh(H).mH @ phi + 0.5 * kraus_map(phi, self.L.mH)
        return out + out.mH
