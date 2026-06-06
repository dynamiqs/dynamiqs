"""Utilitary methods for benchmarking processes.

Classes
-------

* BenchmarkEntry
* MethodParameters

Functions
---------

* extract_method_parameters
* extract_nsteps
* maybe_unpack
"""

from .infos import extract_method_parameters, extract_nsteps
from .structures import BenchmarkEntry, MethodParameters
from .unpacking import maybe_unpack

__all__ = [
    'BenchmarkEntry',
    'MethodParameters',
    'extract_method_parameters',
    'extract_nsteps',
    'maybe_unpack',
]
