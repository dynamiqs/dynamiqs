"""A/B comparison of two benchmark JSON result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_KEY_WIDTH = 60  # kept in sync with `runner._KEY_WIDTH`
_TABLE_HEADER = (
    f'{"case":<{_KEY_WIDTH}} {"before (s)":>11} {"after (s)":>11} {"Δ%":>7}'
    f' {"compile (s)":>15} {"nsteps":>13}'
)


def _key(record: dict[str, Any]) -> tuple[str, str]:
    return record['name'], json.dumps(record['params'], sort_keys=True)


def _case_key(record: dict[str, Any]) -> str:
    params = ','.join(f'{k}={v}' for k, v in sorted(record['params'].items()))
    return f'{record["name"]}[{params}]'


def _format_row(a: dict[str, Any] | None, b: dict[str, Any] | None) -> str:
    key = _case_key(a if a is not None else b)

    def fmt(record: dict[str, Any] | None, field: str, spec: str) -> str:
        if record is None or record[field] is None:
            return '-'
        return format(record[field], spec)

    delta = '-'
    if a is not None and b is not None:
        delta = f'{100 * (b["median_s"] - a["median_s"]) / a["median_s"]:+.1f}'

    compile_ab = f'{fmt(a, "compile_s", ".2f")} → {fmt(b, "compile_s", ".2f")}'
    nsteps_ab = f'{fmt(a, "nsteps", "d")} → {fmt(b, "nsteps", "d")}'
    return (
        f'{key:<{_KEY_WIDTH}} {fmt(a, "median_s", ".4f"):>11}'
        f' {fmt(b, "median_s", ".4f"):>11}'
        f' {delta:>7} {compile_ab:>15} {nsteps_ab:>13}'
    )


def compare(path_a: str | Path, path_b: str | Path) -> None:
    """Print an aligned comparison table of two benchmark result files.

    Rows are matched on (name, params), listed in the first file's order, with
    cases only present in the second file appended. Δ% is the change in median
    run time (positive = slower).
    """
    data_a = json.loads(Path(path_a).read_text())
    data_b = json.loads(Path(path_b).read_text())
    meta_a, meta_b = data_a['meta'], data_b['meta']

    for field in ('precision', 'quick'):
        if meta_a[field] != meta_b[field]:
            print(f'warning: {field} differs: {meta_a[field]} vs {meta_b[field]}')
    if meta_a['device']['platform'] != meta_b['device']['platform']:
        device_a, device_b = meta_a['device']['platform'], meta_b['device']['platform']
        print(f'warning: device differs: {device_a} vs {device_b}')

    print(f'before: {meta_a["git_sha"]} ({meta_a["timestamp"]})')
    print(f'after:  {meta_b["git_sha"]} ({meta_b["timestamp"]})')
    print(_TABLE_HEADER)

    records_a = {_key(r): r for r in data_a['results']}
    records_b = {_key(r): r for r in data_b['results']}
    for key, a in records_a.items():
        print(_format_row(a, records_b.get(key)))
    for key, b in records_b.items():
        if key not in records_a:
            print(_format_row(None, b))
