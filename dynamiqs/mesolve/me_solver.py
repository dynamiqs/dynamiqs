from functools import lru_cache

from torch import Tensor

from ..solvers.solver import Solver
from ..utils.solver_utils import kraus_map


class MESolver(Solver):
    def __init__(self, *args, jump_ops: Tensor):
        super().__init__(*args)
        self.jump_ops = jump_ops[None, ...]  # (1, len(jump_ops), n, n)

    @lru_cache(maxsize=1)
    def sum_no_jump(self) -> Tensor:
        # -> (n, n)
        return (self.jump_ops.adjoint() @ self.jump_ops).squeeze(0).sum(dim=0)

    @lru_cache(maxsize=1)
    def H_nh(self, H: Tensor) -> Tensor:
        # non-hermitian Hamiltonian
        # -> (b_H, 1, n, n)
        return H - 0.5j * self.sum_no_jump()

    def lindbladian(self, t: float, rho: Tensor) -> Tensor:
        H = self.H(t)
        H_nh_rho = self.H_nh(H) @ rho  # (b_H, b_rho, n, n)
        L_rho_Ldag = kraus_map(rho, self.jump_ops)  # (b_H, b_rho, n, n)
        return -1j * (H_nh_rho - H_nh_rho.adjoint()) + L_rho_Ldag
