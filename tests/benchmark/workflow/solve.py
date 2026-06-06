"""Measured problems solving (i.e. computation time, memory...).

Functions
---------

* time_solving
"""

import time

from dynamiqs.result import SolveResult

from ..problems import Problem


def time_solving(problem: Problem) -> tuple[SolveResult, float]:
    """Run a solver over a problem and measure its computation time.

    Params:
        problem: The problem to solve.

    Returns:
        A tuple returning : the solution returned by the solver and the computation time
        of the solving in seconds.
    """
    start_time = time.perf_counter()

    result = problem.solve()

    end_time = time.perf_counter()

    return result, end_time - start_time


__all__ = ['time_solving']
