"""Main logic to run during benchmark processes.

Functions
---------

* main
"""

from pprint import pp

from .problems import CrossResonanceModulatedSESolveProblem
from .workflow import run_benchmark


def main() -> None:
    problem = CrossResonanceModulatedSESolveProblem()

    metrics = run_benchmark(problem)

    # NOTE: this is just for testing for now
    pp(metrics)


if __name__ == '__main__':
    main()


__all__ = ['main']
