from torch import Tensor

from ..solvers.ode.fixed_solver import FixedSolver
from ..utils.solver_utils import lindbladian


class MEEuler(FixedSolver):
    def __init__(self, *args, jump_ops: Tensor):
        super().__init__(*args)
        self.jump_ops = jump_ops  # (len(jump_ops), n, n)

    def forward(self, t: float, dt: float, rho: Tensor) -> Tensor:
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)
        return rho + dt * lindbladian(rho, self.H(t), self.jump_ops)
