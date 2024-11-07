from ..core.abstract_integrator import SESolveIntegrator
from ..core.expm_integrator import SEExpmIntegrator
from ..core.save_mixin import SolveSaveMixin


class SESolveExpmIntegrator(SEExpmIntegrator, SESolveIntegrator, SolveSaveMixin):
    """Integrator computing the time evolution of the Schrödinger equation by
    explicitly exponentiating the propagator.
    """
