from .forward_solver import ForwardSolver


class Euler(ForwardSolver):
    GRADIENT_ALG = ['autograd']
