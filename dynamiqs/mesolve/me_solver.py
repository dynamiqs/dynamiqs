from torch import Tensor

from ..solvers.solver import Solver
from ..utils.solver_utils import cache, kraus_map


class MESolver(Solver):
    def __init__(self, *args, jump_ops: Tensor):
        super().__init__(*args)
        self.jump_ops = jump_ops[None, ...]  # (1, len(jump_ops), n, n)
        self.sum_no_jump = (
            (self.jump_ops.mH @ self.jump_ops).squeeze(0).sum(dim=0)
        )  # (n, n)

        # define cached operator
        # non-hermitian Hamiltonian
        self.H_nh = cache(lambda H: H - 0.5j * self.sum_no_jump)  # (b_H, 1, n, n)

    def lindbladian(self, t: float, rho: Tensor) -> Tensor:
        H = self.H(t)
        H_nh_rho = self.H_nh(H) @ rho  # (b_H, b_rho, n, n)
        L_rho_Ldag = kraus_map(rho, self.jump_ops)  # (b_H, b_rho, n, n)
        return -1j * (H_nh_rho - H_nh_rho.mH) + L_rho_Ldag
