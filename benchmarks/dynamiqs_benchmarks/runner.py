from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import dynamiqs as dq
from dynamiqs import method as dq_method
from dynamiqs.qarrays.qarray import QArray

from .cases import BenchmarkCase, Profile, benchmark_cases


@dataclass(frozen=True)
class MethodSpec:
    name: str
    factory: Any
    supports: tuple[str, ...]
    expm_safe: bool = True


def method_specs(
    dt: float, rtol: float, atol: float, max_steps: int
) -> list[MethodSpec]:
    adaptive = {'rtol': rtol, 'atol': atol, 'max_steps': max_steps}
    return [
        MethodSpec(
            'Tsit5', lambda: dq_method.Tsit5(**adaptive), ('sesolve', 'mesolve')
        ),
        MethodSpec(
            'Dopri5', lambda: dq_method.Dopri5(**adaptive), ('sesolve', 'mesolve')
        ),
        MethodSpec(
            'Dopri8', lambda: dq_method.Dopri8(**adaptive), ('sesolve', 'mesolve')
        ),
        MethodSpec(
            'Kvaerno3', lambda: dq_method.Kvaerno3(**adaptive), ('sesolve', 'mesolve')
        ),
        MethodSpec(
            'Kvaerno5', lambda: dq_method.Kvaerno5(**adaptive), ('sesolve', 'mesolve')
        ),
        MethodSpec('Euler', lambda: dq_method.Euler(dt=dt), ('sesolve', 'mesolve')),
        MethodSpec('Rouchon1', lambda: dq_method.Rouchon1(dt=dt), ('mesolve',)),
        MethodSpec('Rouchon2', lambda: dq_method.Rouchon2(**adaptive), ('mesolve',)),
        MethodSpec('Rouchon3', lambda: dq_method.Rouchon3(**adaptive), ('mesolve',)),
        MethodSpec('Expm', dq_method.Expm, ('sesolve', 'mesolve')),
    ]


def run_suite(
    *,
    output_dir: Path,
    profile: Profile = 'standard',
    selected_cases: set[str] | None = None,
    selected_methods: set[str] | None = None,
    precision: str = 'double',
    warmup: bool = True,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_steps: int = 100_000,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dq.set_precision(precision)  # type: ignore[arg-type]
    dq.set_progress_meter(False)
    rows: list[dict[str, Any]] = []
    metadata = collect_metadata(profile=profile, precision=precision)

    for case in benchmark_cases(profile):
        if selected_cases and case.name not in selected_cases:
            continue
        ref = build_reference(case, profile)
        dt = float(
            (case.tsave[-1] - case.tsave[0])
            / ({'smoke': 24, 'standard': 200, 'full': 500}[profile])
        )
        specs = method_specs(dt=dt, rtol=rtol, atol=atol, max_steps=max_steps)
        for spec in specs:
            if selected_methods and spec.name not in selected_methods:
                continue
            explicit_method = (
                selected_methods is not None and spec.name in selected_methods
            )
            skip_reason = skip_reason_for(case, spec, explicit_method=explicit_method)
            if case.kind not in spec.supports:
                skip_reason = 'unsupported solver/problem combination'
            if skip_reason is not None:
                rows.append(skipped_row(case, spec, metadata, skip_reason))
                continue
            row = run_one(case, spec, ref, metadata, warmup=warmup)
            rows.append(row)

    write_results(output_dir, rows, metadata)
    return rows


def skip_reason_for(
    case: BenchmarkCase, spec: MethodSpec, *, explicit_method: bool = False
) -> str | None:
    if spec.name == 'Expm' and 'time-dependent' in case.tags:
        return 'Expm only supports constant or piecewise-constant generators'
    if (
        not explicit_method
        and spec.name.startswith('Kvaerno')
        and case.kind == 'mesolve'
    ):
        return 'implicit Kvaerno mesolve runs are opt-in because they can be very slow'
    if (
        not explicit_method
        and spec.name.startswith('Kvaerno')
        and 'many-body' in case.tags
    ):
        return 'implicit Kvaerno methods are not informative for this non-stiff scaling case'
    return None


def solve_case(case: BenchmarkCase, method: object):
    kwargs = {
        'exp_ops': case.exp_ops,
        'method': method,
        'progress_meter': False,
        'save_states': case.reference_kind == 'state',
    }
    if case.kind == 'sesolve':
        return dq.sesolve(case.H, case.y0, case.tsave, **kwargs)
    return dq.mesolve(case.H, case.jump_ops, case.y0, case.tsave, **kwargs)


def build_reference(case: BenchmarkCase, profile: Profile) -> dict[str, Any]:
    if case.reference_expect is not None:
        return {
            'kind': 'expect',
            'target': jnp.asarray(case.reference_expect),
            'solver': 'analytical',
        }
    if case.name == 'batched_kerr_oscillator_mesolve' and profile != 'full':
        method = dq_method.Expm()
        solver_name = 'Expm'
    elif case.name == 'reduced_zeno_cnot_mesolve':
        method = dq_method.Rouchon3(
            rtol=1e-10, atol=1e-10, safety_factor=0.75, max_steps=1_000_000
        )
        solver_name = 'Rouchon3-tight'
    else:
        method = dq_method.Dopri8(
            rtol=1e-10, atol=1e-10, safety_factor=0.75, max_steps=1_000_000
        )
        solver_name = 'Dopri8-tight'
    result = solve_case(case, method).block_until_ready()
    target = (
        result.expects if case.reference_kind == 'expect' else to_array(result.states)
    )
    return {
        'kind': case.reference_kind,
        'target': jnp.asarray(target),
        'solver': solver_name,
    }


def run_one(
    case: BenchmarkCase,
    spec: MethodSpec,
    ref: dict[str, Any],
    metadata: dict[str, Any],
    *,
    warmup: bool,
) -> dict[str, Any]:
    method = spec.factory()
    row = base_row(case, spec, metadata)
    row['reference_solver'] = ref['solver']
    try:
        if warmup:
            solve_case(case, method).block_until_ready()
        tic = time.perf_counter()
        result = solve_case(case, method).block_until_ready()
        runtime = time.perf_counter() - tic
        value = result.expects if ref['kind'] == 'expect' else to_array(result.states)
        error = relative_error(jnp.asarray(value), ref['target'])
        row.update(
            status='pass',
            runtime_s=runtime,
            error=error,
            nsteps=extract_nsteps(result.infos),
            naccepted=extract_attr(result.infos, 'naccepted'),
            nrejected=extract_attr(result.infos, 'nrejected'),
            method_settings=str(method),
            message='',
        )
    except Exception as exc:  # benchmark rows should record solver failures
        row.update(
            status='fail',
            runtime_s=np.nan,
            error=np.nan,
            nsteps=np.nan,
            naccepted=np.nan,
            nrejected=np.nan,
            method_settings=str(method),
            message=f'{type(exc).__name__}: {exc}',
        )
    return row


def relative_error(value, target) -> float:
    value_np = np.asarray(jax.device_get(value))
    target_np = np.asarray(jax.device_get(target))
    denom = max(float(np.linalg.norm(target_np.ravel())), 1e-15)
    return float(np.linalg.norm((value_np - target_np).ravel()) / denom)


def to_array(x):
    if isinstance(x, QArray):
        return x.to_jax()
    return dq.to_jax(x)


def extract_nsteps(infos) -> float:
    return extract_attr(infos, 'nsteps')


def extract_attr(infos, attr: str) -> float:
    if infos is None or not hasattr(infos, attr):
        return np.nan
    value = np.asarray(jax.device_get(getattr(infos, attr)))
    return float(value.mean())


def base_row(
    case: BenchmarkCase, spec: MethodSpec, metadata: dict[str, Any]
) -> dict[str, Any]:
    return {
        'benchmark': case.name,
        'kind': case.kind,
        'solver': spec.name,
        'status': 'pending',
        'runtime_s': np.nan,
        'nsteps': np.nan,
        'naccepted': np.nan,
        'nrejected': np.nan,
        'error': np.nan,
        'reference_solver': '',
        'reference_strategy': case.reference_strategy,
        'method_settings': '',
        'profile': metadata['profile'],
        'precision': metadata['precision'],
        'jax_platform': metadata['jax_platform'],
        'tags': ';'.join(case.tags),
        'message': '',
    }


def skipped_row(
    case: BenchmarkCase, spec: MethodSpec, metadata: dict[str, Any], message: str
) -> dict[str, Any]:
    row = base_row(case, spec, metadata)
    row.update(status='skip', message=message)
    return row


def collect_metadata(*, profile: str, precision: str) -> dict[str, Any]:
    return {
        'profile': profile,
        'precision': precision,
        'dynamiqs_version': getattr(dq, '__version__', 'unknown'),
        'jax_version': jax.__version__,
        'jax_platform': jax.default_backend(),
        'python': platform.python_version(),
        'platform': platform.platform(),
    }


def write_results(
    output_dir: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    result_path = output_dir / 'results.csv'
    fieldnames = list(
        base_row(
            benchmark_cases('smoke')[0], MethodSpec('Tsit5', None, ()), metadata
        ).keys()
    )
    with result_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / 'metadata.json').open('w') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    write_leaderboard(output_dir / 'leaderboard.csv', rows)


def write_leaderboard(path: Path, rows: list[dict[str, Any]]) -> None:
    passed = [r for r in rows if r['status'] == 'pass']
    passed.sort(
        key=lambda r: (r['benchmark'], float(r['error']), float(r['runtime_s']))
    )
    fields = [
        'benchmark',
        'solver',
        'runtime_s',
        'error',
        'nsteps',
        'reference_solver',
        'profile',
    ]
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in passed:
            writer.writerow({k: row[k] for k in fields})


def parse_csv_list(values: Iterable[str] | None) -> set[str] | None:
    if not values:
        return None
    parsed = {item for value in values for item in value.split(',') if item}
    return parsed or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run Dynamiqs solver benchmarks.')
    parser.add_argument('--output-dir', type=Path, default=Path('benchmark_results'))
    parser.add_argument(
        '--profile', choices=['smoke', 'standard', 'full'], default='standard'
    )
    parser.add_argument(
        '--case',
        action='append',
        help='Case name(s), comma-separated. Defaults to all.',
    )
    parser.add_argument(
        '--method',
        action='append',
        help='Method name(s), comma-separated. Defaults to all deterministic methods.',
    )
    parser.add_argument('--precision', choices=['single', 'double'], default='double')
    parser.add_argument(
        '--no-warmup',
        action='store_true',
        help='Include first-call compilation in timing.',
    )
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--max-steps', type=int, default=100_000)
    args = parser.parse_args(argv)
    rows = run_suite(
        output_dir=args.output_dir,
        profile=args.profile,
        selected_cases=parse_csv_list(args.case),
        selected_methods=parse_csv_list(args.method),
        precision=args.precision,
        warmup=not args.no_warmup,
        rtol=args.rtol,
        atol=args.atol,
        max_steps=args.max_steps,
    )
    failures = sum(row['status'] == 'fail' for row in rows)
    passes = sum(row['status'] == 'pass' for row in rows)
    skips = sum(row['status'] == 'skip' for row in rows)
    print(
        f'Wrote benchmark results to {args.output_dir} ({passes} pass, {failures} fail, {skips} skip).'
    )
    return 1 if passes == 0 else 0


if __name__ == '__main__':
    raise SystemExit(main())
