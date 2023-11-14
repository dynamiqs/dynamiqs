from __future__ import annotations

from torch import Tensor


class Gradient:
    pass


class Autograd(Gradient):
    pass


class Adjoint(Gradient):
    def __init__(self, *, params: tuple[Tensor, ...]):
        # params (tuple of Tensor): Parameters with respect to which gradients are
        # computed during the adjoint state backward pass.
        self.params = params
