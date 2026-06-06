"""Main workflow steps of the benchmark.

Functions
---------

* run_benchmark
* time_solving
"""

from .benchmark import run_benchmark
from .solve import time_solving

__all__ = ['run_benchmark', 'time_solving']
