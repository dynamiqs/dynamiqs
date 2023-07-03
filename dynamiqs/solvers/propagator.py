from abc import abstractmethod

from torch import Tensor

from ..utils.progress_bar import tqdm
from ..utils.td_tensor import ConstantTDTensor
from .solver import AutogradSolver


class Propagator(AutogradSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check that Hamiltonian is time-independent
        if not isinstance(self.H, ConstantTDTensor):
            raise TypeError(
                'Propagator solvers require a time-independent Hamiltonian.'
            )

    def run_autograd(self):
        y, t1 = self.y0, 0.0
        for t2 in tqdm(self.t_stop.cpu().numpy(), disable=not self.options.verbose):
            y = self.forward(t1, t2 - t1, y)
            self.save(y)
            t1 = t2

    @abstractmethod
    def forward(self, t: float, delta_t: float, y: Tensor):
        pass
