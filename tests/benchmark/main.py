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

from typing import Any
import time as time_module

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

import dynamiqs as dq
from dynamiqs.method import (
    Dopri5,
    Dopri8,
    Expm,
    Tsit5,
)

from problems import (
    ClosedTwoQubit,
    DrivenDampedHarmonicOscillator,
    LargeScaleClosed,
    OpenTimeDependentQubit,
    TimeDependentQubit,
)
from runner import BenchmarkResult, BenchmarkRunner


def rel_error(a: Any, b: Any) -> float:
    if a is None or b is None:
        return 0.0
    a_arr = a.to_jax() if hasattr(a, 'to_jax') else a
    b_arr = b.to_jax() if hasattr(b, 'to_jax') else b
    norm_a = jnp.linalg.norm(a_arr)
    if norm_a < 1e-15:
        return float(jnp.linalg.norm(a_arr - b_arr))
    return float(jnp.linalg.norm(a_arr - b_arr) / norm_a)


def compute_reference(problem) -> tuple[Any, Any]:
    """Compute reference solution for the problem using high-accuracy solver."""
    if hasattr(problem, 'Ls'):
        try:
            ref_result = dq.mesolve(
                problem.H, problem.Ls, problem.y0, problem.tsave,
                exp_ops=problem.Es, method=Expm(), progress_meter=False
            )
        except (TypeError, ValueError):
            ref_result = dq.mesolve(
                problem.H, problem.Ls, problem.y0, problem.tsave,
                exp_ops=problem.Es, method=Tsit5(rtol=1e-8, atol=1e-8), progress_meter=False
            )
        return ref_result.states, ref_result.expects
    else:
        try:
            ref_result = dq.sesolve(
                problem.H, problem.y0, problem.tsave,
                exp_ops=problem.Es, method=Expm(), progress_meter=False
            )
        except (TypeError, ValueError):
            ref_result = dq.sesolve(
                problem.H, problem.y0, problem.tsave,
                exp_ops=problem.Es, method=Tsit5(rtol=1e-8, atol=1e-8), progress_meter=False
            )
        return ref_result.states, ref_result.expects


def main() -> None:
    runner = BenchmarkRunner()
    problems = [
        (ClosedTwoQubit(), [Tsit5(), Dopri5(), Expm()]),
        (DrivenDampedHarmonicOscillator(), [Tsit5(), Dopri5()]),
        (LargeScaleClosed(), [Tsit5(), Dopri5()]),
        (TimeDependentQubit(), [Tsit5(), Dopri8(rtol=1e-8)]),
        (OpenTimeDependentQubit(), [Tsit5()]),
    ]

    for problem, methods in problems:
        print(f'Running benchmark: {problem.name}...', flush=True)

        ref_states, ref_expects = compute_reference(problem)

        for method_cls in methods:
            method_name = (
                method_cls.__name__
                if isinstance(method_cls, type)
                else type(method_cls).__name__
            )

            t0 = time_module.perf_counter()
            nsteps = 0
            state_err = 0.0
            expect_err = 0.0
            status = 'pass'

            try:
                if hasattr(problem, 'Ls'):
                    result = dq.mesolve(
                        problem.H, problem.Ls, problem.y0, problem.tsave,
                        exp_ops=problem.Es, method=method_cls, progress_meter=False
                    )
                else:
                    result = dq.sesolve(
                        problem.H, problem.y0, problem.tsave,
                        exp_ops=problem.Es, method=method_cls, progress_meter=False
                    )
                t0 = time_module.perf_counter() - t0
                nsteps = int(getattr(result.infos, 'nsteps', 0))

                state_err = rel_error(result.states, ref_states)
                expect_err = rel_error(result.expects, ref_expects)
            except (TypeError, ValueError) as e:
                t0 = time_module.perf_counter() - t0
                status = f'fail: {e!r}'[:50]

            runner.results.append(
                BenchmarkResult(
                    problem=problem.name,
                    method=method_name,
                    runtime_s=t0,
                    nsteps=nsteps,
                    state_error=state_err,
                    expect_error=expect_err,
                    status=status,
                )
            )
            print(f'  {method_name}: {t0:.4f}s, state_err={state_err:.2e}', flush=True)

    runner.save_csv()
    runner.print_leaderboard()


if __name__ == '__main__':
    main()