"""Data structures and typing utilities used within the `benchmark` namespace.

Classes
-------

* BenchmarkEntry
* MethodParameters
"""

from typing import TypedDict


class MethodParameters(TypedDict):
    """Descriptiion of a Method configuration."""

    method_name: str
    method_atol: float
    method_rtol: float
    method_factor_min: float
    method_factor_max: float
    method_factor_safety: float


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


__all__ = ['BenchmarkEntry', 'MethodParameters']
