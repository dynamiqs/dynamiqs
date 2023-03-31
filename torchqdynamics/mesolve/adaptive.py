from torch import Tensor

from ..odeint import ForwardQSolver
from ..solver_options import SolverOption
from ..solver_utils import lindbladian
from ..types import TDOperator


class MEAdaptive(ForwardQSolver):
    def __init__(self, H: TDOperator, jump_ops: Tensor, solver_options: SolverOption):
        self.H = H[:, None, ...]  # (b_H, 1, n, n)
        self.jump_ops = jump_ops[None, ...]  # (1, len(jump_ops), n, n)
        self.sum_nojump = (jump_ops.adjoint() @ jump_ops).sum(dim=0)  # (n, n)
        self.options = solver_options

    def forward(self, t: float, rho: Tensor) -> Tensor:
        """Compute drho / dt = L(rho) at time t."""
        return lindbladian(rho, self.H, self.jump_ops)
