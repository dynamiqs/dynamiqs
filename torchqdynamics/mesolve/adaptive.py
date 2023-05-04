from torch import Tensor

from ..solvers.adaptive_solver import AdaptiveSolver
from ..utils.solver_utils import kraus_map


class MEAdaptive(AdaptiveSolver):
    def __init__(self, *args, jump_ops: Tensor):
        super().__init__(*args)

        self.jump_ops = jump_ops[None, ...]  # (1, len(jump_ops), n, n)
        self.sum_nojump = (jump_ops.adjoint() @ jump_ops).sum(dim=0)  # (n, n)

    def odefun(self, t: float, rho: Tensor) -> Tensor:
        """Compute drho / dt = L(rho) at time t."""
        # non-hermitian Hamiltonian
        H_nh = self.H(t) - 0.5j * self.sum_nojump

        # compute Lindblad(t) @ rho
        H_nh_rho = H_nh @ rho
        L_rho_Ldag = kraus_map(rho, self.jump_ops)
        return -1j * (H_nh_rho - H_nh_rho.adjoint()) + L_rho_Ldag
