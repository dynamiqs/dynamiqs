"""Main logic to run during benchmark processes."""

import time
from collections.abc import Callable
from pprint import pp
from typing import TypeAlias, TypedDict

import diffrax
import jax
from cross_resonance_modulated_sesolve import build_problem as cross_resonance

import dynamiqs as dq
from dynamiqs.method import _DEAdaptiveStep
from tests.benchmark import cross_resonance_modulated_sesolve

SEProblem: TypeAlias = Callable[[], tuple[dq.TimeQArray, dq.QArray, jax.Array]]

class BenchmarkEntry(TypedDict):
    """Description of a benchmark record.

    All benchmark record contain information about:
        - The problem that has been solved
        - The method used for the solving and its parameters
        - The precision of the solving, compared to a gold standard
        - The processing speed of the solving (steps, duration...)
        - The engine used for operations (cpu, gpu, tpu)
        - The version used for each backend module (Dynamiqs, JAX...)
    """

    problem_name: str
    method_name: str
    method_atol: float
    method_rtol: float
    method_factor_min: float
    method_factor_max: float
    method_factor_safety: float
    error: float
    nsteps: int
    computation_time: float
    processing_backend: str
    dynamics_version: str
    diffrax_version: str
    jax_version: str


def maybe_unpack(value: float | jax.Array) -> float:
    """If the value is stored in a Jax array, unpack it. Return the value afterwards.

    Params:
        value: Any value that may be packed in a array.

    Returns:
        The value, unpacked if it was stored in an array.
    """
    if isinstance(value, jax.Array):
        return value.item()

    return value


def extract_nsteps(result: dq.result.Result) -> int:
    """Extract the number of steps a solving result has taken.

    If the value is not available, return -1.

    Params
        result: A solving result.

    Returns:
        The number of steps of the solving solution if it is available. -1 otherwise.
    """
    nsteps = getattr(result.infos, "nsteps", -1)

    if isinstance(nsteps, int):
        return nsteps

    if not isinstance(nsteps, jax.Array):
        return -1

    return nsteps.item()


def solve_schrodinger(problem: SEProblem) -> tuple[dq.SESolveResult, float]:
    """Solve the Schrodïnger problem given as parameter.

    Params:
        problem: A function constructing a solvable problem. It should return in order
                  the Hamiltonian, the initial state and the times at which states and
                  expectation values should be saved.

    Returns:
        A tuple returning : the solution returned the solver and the computation time of
        the solving in seconds.
    """
    hamiltonian, initial_state, tsaves = problem()
    start_time = time.perf_counter()

    result = dq.sesolve(
        hamiltonian,
        initial_state,
        tsaves,
        progress_meter=False,
    )

    result.block_until_ready()

    end_time = time.perf_counter()

    return result, end_time - start_time


def extract_schrodinger_metrics(
    result: dq.result.SolveResult,
    problem_name: str,
    computation_time: float,
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
    method_rtol = -1.
    method_atol = -1.
    method_factor_safety = -1.
    method_factor_min = -1.
    method_factor_max = -1.
    error = 0.0
    nsteps = extract_nsteps(result)

    if isinstance(method, _DEAdaptiveStep):
        method_atol = maybe_unpack(method.atol)
        method_rtol = maybe_unpack(method.rtol)
        method_factor_safety = maybe_unpack(method.safety_factor)
        method_factor_min = maybe_unpack(method.min_factor)
        method_factor_max = maybe_unpack(method.max_factor)

    return {
        "problem_name": problem_name,
        "method_name": method_name,
        "method_atol": method_atol,
        "method_rtol": method_rtol,
        "method_factor_min": method_factor_min,
        "method_factor_max": method_factor_max,
        "method_factor_safety": method_factor_safety,
        "error": error,
        "nsteps": nsteps,
        "computation_time": computation_time,
        "processing_backend": jax.default_backend(),
        "dynamics_version": dq.__version__,
        "diffrax_version": diffrax.__version__,
        "jax_version": jax.__version__,
    }


def main() -> None:
    result, computation_time = solve_schrodinger(cross_resonance)

    metrics = extract_schrodinger_metrics(
        result,
        cross_resonance_modulated_sesolve.__name__,
        computation_time
    )

    # NOTE: this is just for testing for now
    pp(metrics)


if __name__ == '__main__':
    main()


__all__ = ['main']
