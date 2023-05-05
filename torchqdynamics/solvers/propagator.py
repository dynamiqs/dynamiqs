from abc import abstractmethod

from torch import Tensor

from ..utils.progress_bar import tqdm
from .solver import AutogradSolver


class Propagator(AutogradSolver):
    def run_autograd(self):
        y, t1 = self.y0, 0.0
        for t2 in tqdm(self.t_save, disable=not self.options.verbose):
            y = self.forward(t1, t2 - t1, y)
            t1 = t2
            self.save(y)

    @abstractmethod
    def forward(self, t: float, dt: float, y: Tensor):
        pass
