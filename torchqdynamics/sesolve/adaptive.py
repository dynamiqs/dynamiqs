from torch import Tensor

<<<<<<< HEAD
<<<<<<< HEAD
from ..ode.forward_solver import ForwardSolver
=======
from ..ode.ode_forward_solver import ODEForwardSolver
>>>>>>> 78bc0c8 (Reorganize main folders)
=======
from ..ode.forward_solver import ODEForwardSolver
>>>>>>> 9b673fa (Rename ODE files)


class SEAdaptive(ForwardSolver):
    def __init__(self, *args):
        super().__init__(*args)

        self.H = self.H[:, None, ...]  # (b_H, 1, n, n)

    def forward(self, t: float, psi: Tensor) -> Tensor:
        """Compute dpsi / dt = -1j * H(psi) at time t."""
        return -1j * self.H @ psi
