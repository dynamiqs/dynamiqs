from .gradient import Adjoint, Autograd, Gradient


class Solver:
    SUPPORTED_GRADIENT = ()

    def supports_gradient(self, gradient: Gradient) -> bool:
        if len(self.SUPPORTED_GRADIENT) == 0:
            return False
        return isinstance(gradient, self.SUPPORTED_GRADIENT)


class Propagator(Solver):
    SUPPORTED_GRADIENT = (Autograd,)


class _ODEFixedStep(Solver):
    SUPPORTED_GRADIENT = (Autograd,)

    def __init__(self, *, dt: float):
        self.dt = dt


class _ODEAdaptiveStep(Solver):
    SUPPORTED_GRADIENT = (Autograd,)

    def __init__(
        self,
        *,
        atol: float = 1e-8,
        rtol: float = 1e-6,
        max_steps: int = 100_000,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
    ):
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps
        self.safety_factor = safety_factor
        self.min_factor = min_factor
        self.max_factor = max_factor


class Dopri5(_ODEAdaptiveStep):
    pass


class Euler(_ODEFixedStep):
    SUPPORTED_GRADIENT = (Autograd, Adjoint)


class Rouchon1(_ODEFixedStep):
    SUPPORTED_GRADIENT = (Autograd, Adjoint)

    def __init__(self, *, dt: float, normalize: bool = True):
        # normalize: If `True`, the scheme is made trace-preserving (up to machine
        # precision) by renormalizing the Kraus map applied at each time step. Ideal
        # for stiff problems. For time-independent problem the Kraus map is normalized
        # with a matrix square root. For time-dependent problems the Kraus map is
        # normalized with a Cholesky decomposition at every time step.
        # at every step to preserve the trace of the density matrix.
        super().__init__(dt=dt)
        self.normalize = normalize


class Rouchon2(_ODEFixedStep):
    SUPPORTED_GRADIENT = (Autograd, Adjoint)
