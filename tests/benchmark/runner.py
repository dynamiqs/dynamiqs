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
from dataclasses import dataclass
from pathlib import Path

import jax

jax.config.update('jax_enable_x64', True)


@dataclass
class BenchmarkResult:
    problem: str
    method: str
    runtime_s: float
    nsteps: int
    state_error: float
    expect_error: float
    status: str


class BenchmarkProblem:
    name: str
    description: str

    def run(self, method) -> tuple:
        raise NotImplementedError


class BenchmarkRunner:
    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / 'results'
        self.output_dir.mkdir(exist_ok=True)
        self.results: list[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def save_csv(self) -> None:
        csv_path = self.output_dir / 'benchmark_results.csv'
        with csv_path.open('w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'problem', 'method', 'runtime_s', 'nsteps',
                'state_rel_error', 'expect_rel_error', 'status'
            ]
            writer.writerow(header)
            for r in self.results:
                row = [
                    r.problem, r.method, r.runtime_s, r.nsteps,
                    r.state_error, r.expect_error, r.status
                ]
                writer.writerow(row)

    def print_leaderboard(self) -> None:
        print('\n=== Dynamiqs Solver Benchmark Results ===\n')
        for problem_name in sorted({r.problem for r in self.results}):
            problem_results = [r for r in self.results if r.problem == problem_name]
            print(f'Problem: {problem_name}')
            print('=' * 70)
            header = (
                f'{"Method":<16} | {"Runtime (s)":<12} | {"Steps":<8} | '
                f'{"State Error":<12} | {"Expect Error":<12} | {"Status"}'
            )
            print(header)
            print('-' * 70)
            for r in sorted(problem_results, key=lambda x: x.runtime_s):
                state_err_str = f'{r.state_error:.2e}'
                expect_err_str = f'{r.expect_error:.2e}'
                print(
                    f'{r.method:<16} | {r.runtime_s:<12.4f} | '
                    f'{r.nsteps:<8} | {state_err_str:<12} | '
                    f'{expect_err_str:<12} | {r.status}'
                )
            print()