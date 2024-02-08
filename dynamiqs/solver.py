from typing import ClassVar, Literal, Optional

import diffrax as dx
import equinox as eqx
from jaxtyping import Scalar

from .gradient import Adjoint, Autograd, Gradient


class Solver(eqx.Module):
    SUPPORTED_GRADIENT: ClassVar[tuple[Gradient]] = ()

    @classmethod
    def supports_gradient(cls, gradient: Gradient) -> bool:
        return isinstance(gradient, cls.SUPPORTED_GRADIENT)


class Propagator(Solver):
    SUPPORTED_GRADIENT = (Autograd,)


class _ODEFixedStep(Solver):
    SUPPORTED_GRADIENT = (Autograd, Adjoint)

    dt: Scalar


class Euler(_ODEFixedStep):
    pass


class Rouchon1(_ODEFixedStep):
    # normalize: The default scheme is trace-preserving at first order only. This
    # parameter sets the normalisation behaviour:
    # - `None`: The scheme is not normalized.
    # - `'sqrt'`: The Kraus map is renormalized with a matrix square root. Ideal
    #   for stiff problems, recommended for time-independent problems.
    # - `cholesky`: The Kraus map is renormalized at each time step using a Cholesky
    #   decomposition. Ideal for stiff problems, recommended for time-dependent
    #   problems.

    normalize: Optional[Literal['sqrt', 'cholesky']] = None


class Rouchon2(_ODEFixedStep):
    pass


class _ODEAdaptiveStep(Solver):
    SUPPORTED_GRADIENT = (Autograd, Adjoint)

    atol: float = 1e-8
    rtol: float = 1e-6
    safety_factor: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 5.0
    max_steps: int = 100_000


class Dopri5(_ODEAdaptiveStep):
    pass


def _stepsize_controller(solver):
    if isinstance(solver, _ODEFixedStep):
        stepsize_controller = dx.ConstantStepSize()
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
