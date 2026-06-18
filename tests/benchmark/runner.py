# Copyright 2023-2025 Dynamiqs developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar

import jax

jax.config.update('jax_enable_x64', True)


@dataclass
class BenchmarkResult:
    problem: str
    method: str
    runtime_s: float
    nsteps: int
    fidelity_error: float
    state_error: float
    status: str

    CSV_HEADER: ClassVar[list[str]] = [
        'problem',
        'method',
        'runtime_s',
        'nsteps',
        'fidelity_error',
        'state_error',
        'status',
    ]

    def to_row(self) -> list:
        return [getattr(self, f.name) for f in fields(self) if f.name != 'CSV_HEADER']


class BenchmarkRunner:
    def __init__(self, output_dir: str | None = None):
        self.output_dir = (
            Path(output_dir) if output_dir else Path(__file__).parent / 'results'
        )
        self.output_dir.mkdir(exist_ok=True)
        self.results: list[BenchmarkResult] = []

    def save_csv(self) -> None:
        csv_path = self.output_dir / 'benchmark_results.csv'
        with csv_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(BenchmarkResult.CSV_HEADER)
            for r in self.results:
                writer.writerow(r.to_row())

    def print_leaderboard(self) -> None:
        # ruff: noqa: T201 - CLI output is intentionally printed to stdout
        print('\n=== Dynamiqs Solver Benchmark Results ===\n')
        for problem_name in sorted({r.problem for r in self.results}):
            problem_results = [r for r in self.results if r.problem == problem_name]
            print(f'Problem: {problem_name}')
            print('=' * 70)
            header = (
                f'{"Method":<16} | {"Runtime (s)":<12} | {"Steps":<8} | '
                f'{"Fidelity Err":<12} | {"L2 Err":<12} | {"Status"}'
            )
            print(header)
            print('-' * 70)
            for r in sorted(problem_results, key=lambda x: x.runtime_s):
                print(
                    f'{r.method:<16} | {r.runtime_s:<12.4f} | {r.nsteps:<8} | '
                    f'{r.fidelity_error:<12.2e} | {r.state_error:<12.2e} | {r.status}'
                )
            print()
