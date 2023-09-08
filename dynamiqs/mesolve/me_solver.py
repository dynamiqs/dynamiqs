from torch import Tensor

from ..solvers.solver import Solver
from ..solvers.utils import cache, kraus_map


class MESolver(Solver):
    def __init__(self, *args, jump_ops: Tensor):
        super().__init__(*args)
        self.L = jump_ops[None, ...]  # (1, len(L), n, n)
        self.LdagL = (self.L.mH @ self.L).squeeze(0).sum(dim=0)  # (n, n)

        # define cached operator
        # non-hermitian Hamiltonian
        self.H_nh = cache(lambda H: H - 0.5j * self.LdagL)  # (b_H, 1, n, n)

    def lindbladian(self, t: float, rho: Tensor) -> Tensor:
        """Compute the action of the Lindbladian on the density matrix.

        Notes:
            Hermiticity of the output is enforced to avoid numerical instability
            with Runge-Kutta solvers.
        """
        H = self.H(t)
        out = -1j * self.H_nh(H) @ rho + 0.5 * kraus_map(rho, self.L)
        return out + out.mH  # (b_H, b_rho, n, n)
