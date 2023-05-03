import torch

from ..solver import Solver
from ..utils.progress_bar import tqdm


class SEPropagator(Solver):
    def __init__(self, *args):
        super().__init__(*args)

        self.H = self.H[:, None, ...]  # (b_H, 1, n, n)

    def odeint(self):
        y, t0 = self.y0, 0.0
        for i, ti in enumerate(tqdm(self.t_save, disable=not self.options.verbose)):
            y = torch.matrix_exp(-1j * self.H * (ti - t0)) @ y
            t0 = ti
            self.save(i, y)
