from __future__ import annotations

import torch.nn as nn


class Gradient:
    pass


class Autograd(Gradient):
    pass


class Adjoint(Gradient):
    def __init__(self, *, params: tuple[nn.Parameter, ...]):
        # params (tuple of nn.Parameter): Parameters with respect to which
        #     gradients are computed during the adjoint state backward pass.
        self.params = params
