"""Main logic to run during benchmark processes."""

from pprint import pp

from benchmark import run_benchmark
from problems import CrossResonanceModulatedSESolveProblem


def main() -> None:
    problem = CrossResonanceModulatedSESolveProblem()

    metrics = run_benchmark(problem)

    # NOTE: this is just for testing for now
    pp(metrics)


if __name__ == '__main__':
    main()


__all__ = ['main']
