from abc import abstractmethod

import torch

from .solver import Solver
from .utils.progress_bar import tqdm


class Propagator(Solver):
    GRADIENT_ALG = ['autograd']

    def integrate_nograd(self):
        with torch.no_grad():
            self.integrate_autograd()

    def integrate_autograd(self):
        y, t1 = self.y0, 0.0
        for t2 in tqdm(self.t_save, disable=not self.options.verbose):
            y = self.forward(y, t1, t2 - t1)
            t1 = t2
            self.save(y)

    def integrate_adjoint(self):
        return NotImplementedError(
            'This solver does not support adjoint-based gradient computation.'
        )

    @abstractmethod
    def forward(self, t, y, dt):
        pass
