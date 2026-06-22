from __future__ import annotations

# ruff: noqa: T201
import argparse
import csv
import platform
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp

import dynamiqs as dq

from .metrics import extract_nsteps_stats
from .problems import BenchmarkProblem, MethodSpec, all_problems, smoke_problems

Precision = Literal['single', 'double']


@dataclass(frozen=True)
class BenchmarkRow:
    benchmark: str
    kind: str
    method: str
    reference: str
    precision: str
    reference_precision: str
    runtime_s: float | None
    nsteps_mean: float | None
    nsteps_min: float | None
    nsteps_max: float | None
    error_mean: float | None
    error_min: float | None
    error_max: float | None
    status: str
    message: str
    dynamiqs_version: str
    jax_version: str
    backend: str
    platform: str


@contextmanager
def _precision(precision: Precision):
    previous_x64 = jax.config.read('jax_enable_x64')
    dq.set_precision(precision)
    try:
        yield
    finally:
        dq.set_precision('double' if previous_x64 else 'single')


def _metadata() -> dict[str, str]:
    return {
        'dynamiqs_version': dq.__version__,
        'jax_version': jax.__version__,
        'backend': jax.default_backend(),
        'platform': platform.platform(),
    }


def _target_complex_dtype(precision: Precision) -> Any:
    return jnp.complex64 if precision == 'single' else jnp.complex128


def _reference(
    problem: BenchmarkProblem, *, precision: Precision, reference_precision: Precision
) -> Any:
    with _precision(reference_precision):
        reference = dq.to_jax(problem.reference())
        reference = reference.astype(_target_complex_dtype(precision))
        reference.block_until_ready()
        return reference


def _empty_row(
    problem: BenchmarkProblem,
    method: MethodSpec,
    *,
    precision: Precision,
    reference_precision: Precision,
    message: str,
) -> BenchmarkRow:
    return BenchmarkRow(
        benchmark=problem.name,
        kind=problem.kind,
        method=method.name,
        reference=problem.reference_name,
        precision=precision,
        reference_precision=reference_precision,
        runtime_s=None,
        nsteps_mean=None,
        nsteps_min=None,
        nsteps_max=None,
        error_mean=None,
        error_min=None,
        error_max=None,
        status='fail',
        message=message,
        **_metadata(),
    )


def _run_once(
    problem: BenchmarkProblem,
    method_spec: MethodSpec,
    reference: Any,
    *,
    precision: Precision,
    reference_precision: Precision,
) -> BenchmarkRow:
    metadata = _metadata()
    start = time.perf_counter()
    try:
        result = problem.run(method_spec.factory())
        result.block_until_ready()
        runtime_s = time.perf_counter() - start
        error = problem.error(result, reference)
        nsteps = extract_nsteps_stats(result)
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
        method=method_spec.name,
        reference=problem.reference_name,
        precision=precision,
        reference_precision=reference_precision,
        runtime_s=runtime_s,
        nsteps_mean=None if nsteps is None else nsteps.mean,
        nsteps_min=None if nsteps is None else nsteps.minimum,
        nsteps_max=None if nsteps is None else nsteps.maximum,
        error_mean=None if error is None else error.mean,
        error_min=None if error is None else error.minimum,
        error_max=None if error is None else error.maximum,
        status=status,
        message=message,
        **metadata,
    )


def _method_selected(method: MethodSpec, method_filter: set[str] | None) -> bool:
    if method_filter is None:
        return True
    return method.name in method_filter or method.family in method_filter


def run_suite(
    problems: tuple[BenchmarkProblem, ...],
    method_filter: set[str] | None = None,
    *,
    precision: Precision = 'single',
    reference_precision: Precision = 'double',
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    with _precision(precision):
        for problem in problems:
            selected_methods = tuple(
                method
                for method in problem.methods
                if _method_selected(method, method_filter)
            )
            if not selected_methods:
                continue
            try:
                reference = _reference(
                    problem,
                    precision=precision,
                    reference_precision=reference_precision,
                )
            except Exception as exc:  # noqa: BLE001
                message = f'reference failed: {type(exc).__name__}: {exc}'
                rows.extend(
                    _empty_row(
                        problem,
                        method,
                        precision=precision,
                        reference_precision=reference_precision,
                        message=message,
                    )
                    for method in selected_methods
                )
                continue
            for method in selected_methods:
                rows.append(
                    _run_once(
                        problem,
                        method,
                        reference,
                        precision=precision,
                        reference_precision=reference_precision,
                    )
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


def _format_stats(mean: float | None, min_: float | None, max_: float | None) -> str:
    if mean is None or min_ is None or max_ is None:
        return '-'
    return f'{mean:.3g}/{min_:.3g}/{max_:.3g}'


def print_leaderboard(rows: list[BenchmarkRow]) -> None:
    print('\nDynamiqs solver benchmark leaderboard')
    print('-' * 122)
    print(
        f'{"benchmark":34} {"method":20} {"status":6} '
        f'{"runtime_s":>10} {"nsteps mean/min/max":>22} {"error mean/min/max":>24}'
    )
    print('-' * 122)
    for row in rows:
        runtime = '-' if row.runtime_s is None else f'{row.runtime_s:.4f}'
        nsteps = _format_stats(row.nsteps_mean, row.nsteps_min, row.nsteps_max)
        error = _format_stats(row.error_mean, row.error_min, row.error_max)
        print(
            f'{row.benchmark:34} {row.method:20} {row.status:6} '
            f'{runtime:>10} {nsteps:>22} {error:>24}'
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Dynamiqs solver benchmarks.')
    parser.add_argument('--suite', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--out', type=Path, default=Path('benchmark-results.csv'))
    parser.add_argument(
        '--methods',
        nargs='*',
        help='Method names or families, for example Euler or "Euler(dt=1e-03)".',
    )
    parser.add_argument('--precision', choices=('single', 'double'), default='single')
    parser.add_argument(
        '--reference-precision', choices=('single', 'double'), default='double'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problems = smoke_problems() if args.suite == 'smoke' else all_problems()
    method_filter = set(args.methods) if args.methods else None
    rows = run_suite(
        problems,
        method_filter,
        precision=args.precision,
        reference_precision=args.reference_precision,
    )
    write_csv(rows, args.out)
    print_leaderboard(rows)
    print(f'\nSaved benchmark results to {args.out}')


if __name__ == '__main__':
    main()
