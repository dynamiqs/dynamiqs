from __future__ import annotations


class Gradient:
    pass


class AutogradMethod:
    def autograd(self):
        self.gradient = Autograd()
        return self


class Autograd(Gradient):
    pass


class AdjointMethod:
    def adjoint(self, parameters):
        self.gradient = Adjoint(parameters)
        return self


class Adjoint(Gradient):
    NAME = "adjoint"

    def __init__(self, parameters):
        self.parameters = parameters
