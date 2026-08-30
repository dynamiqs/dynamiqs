"""Benchmark runner: timing loop, metadata collection, table and JSON output."""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from dynamiqs.qarrays.layout import get_layout

from .cases import Case, benchmark_cases

_TABLE_HEADER = (
    f'{"case":<40} {"compile (s)":>12} {"median (s)":>12} {"nsteps":>8} {"nrej":>6}'
)


def _extract_infos(out: Any) -> dict[str, int | None]:
    # nsteps/naccepted/nrejected are batched arrays for batched simulations; the
    # slowest batch lane governs wall-clock time, so reduce with max
    infos = getattr(out, 'infos', None)

    def get(name: str) -> int | None:
        value = getattr(infos, name, None)
        return None if value is None else int(jnp.max(value))

    return {
        'nsteps': get('nsteps'),
        'naccepted': get('naccepted'),
        'nrejected': get('nrejected'),
    }


def run_case(case: Case, repeats: int = 5) -> dict[str, Any]:
    """Run a single benchmark case and return its result record.

    The first call is recorded as `compile_s` (compilation + one run, an upper
    bound on compilation time). The same jitted closure is then called `repeats`
    times and the median wall-clock time is reported as `median_s`.
    """
    fn = case.build()

    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    compile_s = time.perf_counter() - t0
    infos = _extract_infos(out)

    runs_s = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        runs_s.append(time.perf_counter() - t0)

    return {
        'name': case.name,
        'params': case.params,
        'compile_s': compile_s,
        'runs_s': runs_s,
        'median_s': statistics.median(runs_s),
        **infos,
    }


def _git_info() -> tuple[str | None, bool | None]:
    def git(*args: str) -> str:
        cwd = Path(__file__).parent
        return subprocess.check_output(
            ['git', *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return git('rev-parse', 'HEAD'), git('status', '--porcelain') != ''
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _metadata(quick: bool, repeats: int, filter_: str | None) -> dict[str, Any]:
    git_sha, git_dirty = _git_info()
    device = jax.devices()[0]
    packages = ['dynamiqs', 'jax', 'jaxlib', 'diffrax', 'equinox']
    return {
        'timestamp': datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds'),
        'git_sha': git_sha,
        'git_dirty': git_dirty,
        'platform': platform.platform(),
        'python': platform.python_version(),
        'device': {'platform': device.platform, 'device_kind': device.device_kind},
        'versions': {p: importlib.metadata.version(p) for p in packages},
        'precision': 'double' if jax.config.jax_enable_x64 else 'single',
        'layout': str(get_layout()),
        'quick': quick,
        'repeats': repeats,
        'filter': filter_,
    }


def _format_row(record: dict[str, Any], key: str) -> str:
    nsteps = record['nsteps']
    nrejected = record['nrejected']
    return (
        f'{key:<40} {record["compile_s"]:>12.3f} {record["median_s"]:>12.4f}'
        f' {nsteps if nsteps is not None else "-":>8}'
        f' {nrejected if nrejected is not None else "-":>6}'
    )


def run_suite(
    quick: bool = False,
    filter_: str | None = None,
    repeats: int = 5,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Run the benchmark suite and return `{'meta': ..., 'results': [...]}`.

    Args:
        quick: If `True`, use tiny problem sizes (for CPU CI and sanity runs).
        filter_: If given, only run cases whose key contains this substring.
        repeats: Number of timed runs per case (after one compile/warmup run).
        out: If given, path of the JSON file to write results to.
    """
    cases = benchmark_cases(quick=quick)
    if filter_ is not None:
        cases = [c for c in cases if filter_ in c.key]

    meta = _metadata(quick, repeats, filter_)
    device = meta['device']
    print(f'device: {device["platform"]} ({device["device_kind"]})', end=', ')
    print(f'precision: {meta["precision"]}, repeats: {repeats}')
    print(_TABLE_HEADER)

    results = []
    for case in cases:
        record = run_case(case, repeats=repeats)
        results.append(record)
        print(_format_row(record, case.key))

    data = {'meta': meta, 'results': results}
    if out is not None:
        Path(out).write_text(json.dumps(data, indent=2) + '\n')
        print(f'results written to {out}')
    return data
