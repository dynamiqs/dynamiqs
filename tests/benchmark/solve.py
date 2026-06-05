"""Measured solving of problems."""

import time

from problems import Problem

from dynamiqs.result import SolveResult


def time_solving(problem: Problem) -> tuple[SolveResult, float]:
    """Measure the time taken by a solver.

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
