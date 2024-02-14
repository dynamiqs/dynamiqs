from __future__ import annotations

from typing import ClassVar

import equinox as eqx
from jaxtyping import Scalar

from ._utils import obj_type_str
from .gradient import Autograd, CheckpointAutograd, Gradient


# === generic solvers options
class Solver(eqx.Module):
    SUPPORTED_GRADIENT: ClassVar[tuple] = ()  # todo: typing
    # Note: The next line is commented due to an issue with dataclasses and the
    # ordering of non-default vs default argument (see e.g.
    # https://stackoverflow.com/a/53085935). This issue is inherited by equinox Module.
    # For simplicity, we skip the definition of gradient in the base class.
    # gradient: Gradient

    def __check_init__(self):
        if not isinstance(self.gradient, self.SUPPORTED_GRADIENT):
            support_str = ', '.join(f'`{x.__name__}`' for x in self.SUPPORTED_GRADIENT)
            raise ValueError(
                f'Solver `{obj_type_str(self)}` does not support gradient'
                f' `{obj_type_str(self.gradient)}` (supported gradient types:'
                f' {support_str}).'
            )


# === propagator solvers options
class Propagator(Solver):
    SUPPORTED_GRADIENT = (Autograd,)
    gradient: Gradient = Autograd()


# === generic ODE solvers options
class _ODESolver(Solver):
    pass


class _ODEFixedStep(_ODESolver):
    dt: Scalar


class _ODEAdaptiveStep(_ODESolver):
    rtol: float = 1e-4
    atol: float = 1e-6
    safety_factor: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 5.0
    max_steps: int = 100_000


# === diffrax-based solvers options
class _DiffraxSolver(Solver):
    SUPPORTED_GRADIENT = (Autograd, CheckpointAutograd)
    gradient: Gradient = CheckpointAutograd()


# === public solvers options
class Euler(_DiffraxSolver, _ODEFixedStep):
    pass


class Rouchon1(_DiffraxSolver, _ODEFixedStep):
    # normalize: The default scheme is trace-preserving at first order only. This
    # parameter sets the normalisation behaviour:
    # - `None`: The scheme is not normalized.
    # - `'sqrt'`: The Kraus map is renormalized with a matrix square root. Ideal
    #   for stiff problems, recommended for time-independent problems.
    # - `cholesky`: The Kraus map is renormalized at each time step using a Cholesky
    #   decomposition. Ideal for stiff problems, recommended for time-dependent
    #   problems.

    # todo: fix, strings are not valid JAX types
    # normalize: Literal['sqrt', 'cholesky'] | None = None
    pass


class Rouchon2(_DiffraxSolver, _ODEFixedStep):
    pass


class Dopri5(_DiffraxSolver, _ODEAdaptiveStep):
    pass


class Dopri8(_DiffraxSolver, _ODEAdaptiveStep):
    pass


class Tsit5(_DiffraxSolver, _ODEAdaptiveStep):
    pass
