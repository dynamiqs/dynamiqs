"""Base, common structures for all concrete problems.

See the `concrete` sibling namespace for concrete problems used in the
benchmarking.

Classes
-------

* Problem
* SESolveProblem
"""

from .problem import Problem
from .sesolve_problem import SESolveProblem

__all__ = ['Problem', 'SESolveProblem']
