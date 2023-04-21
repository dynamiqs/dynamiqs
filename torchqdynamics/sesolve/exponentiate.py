import torch

from ..progress_bar import tqdm
from ..qsolver import QSolver
from ..tensor_types import TDOperator


class SEExponentiate(QSolver):
    def __init__(self, *args, H: TDOperator):
        super().__init__(*args)

        # convert H to size compatible with (b_H, b_psi, n, n)
        self.H = H[:, None, ...]

    def run(self):
        y, t1 = self.y0, 0.0
        for t2 in tqdm(self.t_save, disable=not self.options.verbose):
            y = torch.matrix_exp(-1j * self.H * (t2 - t1)) @ y
            t1 = t2
            self.save(y)
