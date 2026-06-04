from __future__ import annotations

# ruff: noqa: T201
import argparse
import csv
import platform
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import jax

import dynamiqs as dq
from dynamiqs.method import (
    Dopri5,
    Dopri8,
    Euler,
    Expm,
    Kvaerno3,
    Kvaerno5,
    Method,
    Rouchon1,
    Rouchon2,
    Rouchon3,
    Tsit5,
)

from .metrics import extract_nsteps
from .problems import BenchmarkProblem, all_problems, smoke_problems

METHODS: dict[str, Method] = {
    'Tsit5': Tsit5(),
    'Dopri5': Dopri5(),
    'Dopri8': Dopri8(),
    'Kvaerno3': Kvaerno3(),
    'Kvaerno5': Kvaerno5(),
    'Euler': Euler(dt=1e-3),
    'Rouchon1': Rouchon1(dt=1e-3),
    'Rouchon2': Rouchon2(),
    'Rouchon3': Rouchon3(),
    'Expm': Expm(),
}


@dataclass(frozen=True)
class BenchmarkRow:
    benchmark: str
    kind: str
    method: str
    reference: str
    runtime_s: float | None
    nsteps: float | None
    error: float | None
    status: str
    message: str
    dynamiqs_version: str
    jax_version: str
    backend: str
    platform: str


def _metadata() -> dict[str, str]:
    return {
        'dynamiqs_version': dq.__version__,
        'jax_version': jax.__version__,
        'backend': jax.default_backend(),
        'platform': platform.platform(),
    }


def _run_once(
    problem: BenchmarkProblem, method_name: str, method: Method, reference: Any
) -> BenchmarkRow:
    metadata = _metadata()
    start = time.perf_counter()
    try:
        result = problem.run(method)
        result.block_until_ready()
        runtime_s = time.perf_counter() - start
        error = problem.error(result, reference)
        nsteps = extract_nsteps(result)
        status = 'pass'
        message = ''
    except Exception as exc:  # noqa: BLE001 - benchmark should record solver failures.
        runtime_s = None
        error = None
        nsteps = None
        status = 'fail'
        message = f'{type(exc).__name__}: {exc}'

    return BenchmarkRow(
        benchmark=problem.name,
        kind=problem.kind,
        method=method_name,
        reference=problem.reference_name,
        runtime_s=runtime_s,
        nsteps=nsteps,
        error=error,
        status=status,
        message=message,
        **metadata,
    )


def run_suite(
    problems: tuple[BenchmarkProblem, ...], method_filter: set[str] | None = None
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for problem in problems:
        try:
            reference = problem.reference()
        except Exception as exc:  # noqa: BLE001 - benchmark should record reference failures.
            message = f'reference failed: {type(exc).__name__}: {exc}'
            for method_name in problem.deterministic_methods:
                if method_filter is not None and method_name not in method_filter:
                    continue
                rows.append(
                    BenchmarkRow(
                        benchmark=problem.name,
                        kind=problem.kind,
                        method=method_name,
                        reference=problem.reference_name,
                        runtime_s=None,
                        nsteps=None,
                        error=None,
                        status='fail',
                        message=message,
                        **_metadata(),
                    )
                )
            continue
        for method_name in problem.deterministic_methods:
            if method_filter is not None and method_name not in method_filter:
                continue
            rows.append(
                _run_once(problem, method_name, METHODS[method_name], reference)
            )
    return rows


def write_csv(rows: list[BenchmarkRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(BenchmarkRow)]
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def print_leaderboard(rows: list[BenchmarkRow]) -> None:
    print('\nDynamiqs solver benchmark leaderboard')
    print('-' * 92)
    print(
        f'{"benchmark":34} {"method":10} {"status":6} '
        f'{"runtime_s":>10} {"nsteps":>10} {"error":>12}'
    )
    print('-' * 92)
    for row in rows:
        runtime = '-' if row.runtime_s is None else f'{row.runtime_s:.4f}'
        nsteps = '-' if row.nsteps is None else f'{row.nsteps:.1f}'
        error = '-' if row.error is None else f'{row.error:.3e}'
        print(
            f'{row.benchmark:34} {row.method:10} {row.status:6} '
            f'{runtime:>10} {nsteps:>10} {error:>12}'
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Dynamiqs solver benchmarks.')
    parser.add_argument('--suite', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--out', type=Path, default=Path('benchmark-results.csv'))
    parser.add_argument('--methods', nargs='*', help='Optional subset of method names.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problems = smoke_problems() if args.suite == 'smoke' else all_problems()
    method_filter = set(args.methods) if args.methods else None
    rows = run_suite(problems, method_filter)
    write_csv(rows, args.out)
    print_leaderboard(rows)
    print(f'\nSaved benchmark results to {args.out}')


if __name__ == '__main__':
    main()
