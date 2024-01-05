from .gradient import Gradient


class Solver:
    SUPPORTED_GRADIENT = ()

    def supports_gradient(self, gradient: Gradient) -> bool:
        if len(self.SUPPORTED_GRADIENT) == 0:
            return False
        return isinstance(gradient, self.SUPPORTED_GRADIENT)


class Dopri5(Solver):
    pass


class Euler(Solver):
    pass
