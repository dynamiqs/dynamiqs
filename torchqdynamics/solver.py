"""Gather all solver options under the same namespace `solver`."""

from .mesolve.solver_options import Rouchon, Rouchon1, Rouchon1_5, Rouchon2
from .sesolve.solver_options import Propagator
from .solver_options import Dopri45, Euler
