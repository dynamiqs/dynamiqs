import diffrax
import jax

import dynamiqs as dq

from .problems import Problem
from .solve import time_solving
from .utils import BenchmarkEntry, extract_method_parameters, extract_nsteps


def run_benchmark(problem: Problem) -> BenchmarkEntry:
    """Create a benchmark entry based on a Problem.

    Once the benchmark is complete, it will retrieve the following metrics:
        - The name of the problem used for the benchmark entry
        - The method used for the solving and its parameters
        - The precision of the solving, compared to the gold standard defined
          in the problem
        - The processing speed of the solving (steps, duration...)
        - The engine used for operations (cpu, gpu, tpu)
        - The version used for each backend module (Dynamiqs, JAX...)

    Method parameters will be defined assuming it can receive atol, rtol and factor
    parameters. If it cannot, all the values for them in the entry will be set to -1.

    Params:
        problem: A solvable Problem

    Returns:
        A record of metrics, as specified in the description.
    """
    result, computation_time = time_solving(problem)

    method_params = extract_method_parameters(result.method)
    nsteps = extract_nsteps(result)

    # TODO: implement a method to compare with reference for any method
    error = 0.0

    return {
        'problem_name': problem.get_name(),
        'method_name': method_params['method_name'],
        'method_atol': method_params['method_atol'],
        'method_rtol': method_params['method_rtol'],
        'method_factor_min': method_params['method_factor_min'],
        'method_factor_max': method_params['method_factor_max'],
        'method_factor_safety': method_params['method_factor_safety'],
        'error': error,
        'nsteps': nsteps,
        'computation_time': computation_time,
        'processing_backend': jax.default_backend(),
        'dynamics_version': dq.__version__,
        'diffrax_version': diffrax.__version__,
        'jax_version': jax.__version__,
    }


__all__ = ['run_benchmark']
