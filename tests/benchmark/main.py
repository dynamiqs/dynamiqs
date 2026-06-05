"""Main logic to run during benchmark processes."""

import time
from pprint import pp

import diffrax
import jax
from problems import CrossResonanceModulatedSESolveProblem, Problem
from utils import BenchmarkEntry, extract_nsteps, maybe_unpack

import dynamiqs as dq
from dynamiqs.method import _DEAdaptiveStep
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


def extract_schrodinger_metrics(
    result: SolveResult, problem_name: str, computation_time: float
) -> BenchmarkEntry:
    """Construct a record of metrics based on a Result.

    Typically, this retrieves the metrics we want to construct a benchmark row.

    So far, the array of metrics will contain:
        - The name of the method used
        - The parameters of the method (in order): atol, rtol, safety factor, min
          factor, max factor
        - 0.0 (this will in the future contain the accuracy compared to a reference)
        - The number of steps the result has taken (typically, for example, for solvers)
        - The computation time of the operation, in seconds.

    Params:
        result: An operation result.
        computation_time: The operation computation time. It is expected to have been
                          determined externally but is provided to be aggregated into
                          the record.

    Returns:
        An array of metrics, as specified in the description.
    """
    # TODO: implement a method to compare with reference for any method
    method = result.method
    method_name = type(method).__name__
    method_rtol = -1.0
    method_atol = -1.0
    method_factor_safety = -1.0
    method_factor_min = -1.0
    method_factor_max = -1.0
    error = 0.0
    nsteps = extract_nsteps(result)

    if isinstance(method, _DEAdaptiveStep):
        method_atol = maybe_unpack(method.atol)
        method_rtol = maybe_unpack(method.rtol)
        method_factor_safety = maybe_unpack(method.safety_factor)
        method_factor_min = maybe_unpack(method.min_factor)
        method_factor_max = maybe_unpack(method.max_factor)

    return {
        'problem_name': problem_name,
        'method_name': method_name,
        'method_atol': method_atol,
        'method_rtol': method_rtol,
        'method_factor_min': method_factor_min,
        'method_factor_max': method_factor_max,
        'method_factor_safety': method_factor_safety,
        'error': error,
        'nsteps': nsteps,
        'computation_time': computation_time,
        'processing_backend': jax.default_backend(),
        'dynamics_version': dq.__version__,
        'diffrax_version': diffrax.__version__,
        'jax_version': jax.__version__,
    }


def main() -> None:
    problem = CrossResonanceModulatedSESolveProblem()
    result, computation_time = time_solving(problem)

    metrics = extract_schrodinger_metrics(result, problem.get_name(), computation_time)

    # NOTE: this is just for testing for now
    pp(metrics)


if __name__ == '__main__':
    main()


__all__ = ['main']
