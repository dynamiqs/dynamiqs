"""Gather all solver options under the same namespace `solver`."""

from .mesolve.solver_option import Rouchon, Rouchon1, Rouchon1_5, Rouchon2
from .sesolve.solver_option import Propagator
from .solver_option import Dopri45, Euler
