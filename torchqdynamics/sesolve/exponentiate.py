import torch

from ..progress_bar import tqdm
from ..qsolver import QSolver


class SEExponentiate(QSolver):
    def __init__(self, *args):
        super().__init__(*args)

        self.H = self.H[:, None, ...]  # (b_H, 1, n, n)

    def run(self):
        y, t1 = self.y0, 0.0
        for t2 in tqdm(self.t_save, disable=not self.options.verbose):
            y = torch.matrix_exp(-1j * self.H * (t2 - t1)) @ y
            t1 = t2
            self.save(y)
