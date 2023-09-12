from abc import abstractmethod

from torch import Tensor

from .solver import AutogradSolver
from .utils.td_tensor import ConstantTDTensor
from .utils.utils import tqdm


class Propagator(AutogradSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check that Hamiltonian is time-independent
        if not isinstance(self.H, ConstantTDTensor):
            raise TypeError(
                'Solver `Propagator` requires a time-independent Hamiltonian.'
            )
        self.H = self.H(0.0)

    def run_autograd(self):
        y, t1 = self.y0, 0.0
        for t2 in tqdm(self.t_stop.cpu().numpy(), disable=not self.options.verbose):
            y = self.forward(t1, t2 - t1, y)
            self.save(y)
            t1 = t2

    @abstractmethod
    def forward(self, t: float, delta_t: float, y: Tensor):
        pass
