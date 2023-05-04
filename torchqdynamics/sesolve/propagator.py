import torch

from ..solver import Solver
from ..utils.progress_bar import tqdm


class SEPropagator(Solver):
    def __init__(self, *args):
        super().__init__(*args)

    def run(self):
        y, t1 = self.y0, 0.0
        for t2 in tqdm(self.t_save, disable=not self.options.verbose):
            y = torch.matrix_exp(-1j * self.H(t2) * (t2 - t1)) @ y
            t1 = t2
            self.save(y)
