import diffrax as dx
from jax import Array

from .gradient import Adjoint, Autograd, Gradient


class Solver:
    SUPPORTED_GRADIENT = ()

    def supports_gradient(self, gradient: Gradient) -> bool:
        if len(self.SUPPORTED_GRADIENT) == 0:
            return False
        return isinstance(gradient, self.SUPPORTED_GRADIENT)


class _ODEFixedStep(Solver):
    SUPPORTED_GRADIENT = (Autograd,)

    def __init__(self, *, dt: float):
        # convert `dt` in case an array was passed instead of a float
        if isinstance(dt, Array):
            dt = dt.item()
        self.dt = dt


class _ODEAdaptiveStep(Solver):
    SUPPORTED_GRADIENT = (Autograd, Adjoint)

    def __init__(
        self,
        *,
        atol: float = 1e-8,
        rtol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
    ):
        self.atol = atol
        self.rtol = rtol
        self.safety_factor = safety_factor
        self.min_factor = min_factor
        self.max_factor = max_factor


class Dopri5(_ODEAdaptiveStep):
    pass


class Euler(_ODEFixedStep):
    pass


def _stepsize_controller(solver):
    if isinstance(solver, _ODEFixedStep):
        stepsize_controller = dx.ConstantStepSize
        dt = solver.dt
    elif isinstance(solver, _ODEAdaptiveStep):
        stepsize_controller = dx.PIDController(
            rtol=solver.rtol,
            atol=solver.atol,
            safety=solver.safety_factor,
            factormin=solver.min_factor,
            factormax=solver.max_factor,
        )
        dt = None
    else:
        raise RuntimeError(
            'Should never occur, all solvers should be either fixed or adaptive.'
        )
    return stepsize_controller, dt
