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

import time as time_module

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from problems import ALL_PROBLEMS, BenchmarkProblem  # noqa: E402
from runner import BenchmarkResult, BenchmarkRunner  # noqa: E402

import dynamiqs as dq  # noqa: E402
from dynamiqs.method import Dopri8, Expm, Method  # noqa: E402


def _fidelity_error(sim_states, ref_states):
    """Compute 1 - fidelity between simulated and reference states at each save time."""
    if sim_states is None or ref_states is None:
        return float('nan')
    sim = sim_states.to_jax() if hasattr(sim_states, 'to_jax') else sim_states
    ref = ref_states.to_jax() if hasattr(ref_states, 'to_jax') else ref_states
    # Normalise states if they are kets
    sim = sim / jnp.linalg.norm(sim, axis=-2, keepdims=True)
    ref = ref / jnp.linalg.norm(ref, axis=-2, keepdims=True)
    # Fidelity for pure states: |<sim|ref>|^2
    inner = jnp.sum(jnp.conj(sim) * ref, axis=-2)
    fidelities = jnp.abs(inner) ** 2
    return float(jnp.mean(1.0 - fidelities))


def _l2_state_error(sim_states, ref_states):
    """Compute L2 norm of state difference at each save time (normalised)."""
    if sim_states is None or ref_states is None:
        return float('nan')
    sim = sim_states.to_jax() if hasattr(sim_states, 'to_jax') else sim_states
    ref = ref_states.to_jax() if hasattr(ref_states, 'to_jax') else ref_states
    diff = jnp.linalg.norm(sim - ref, axis=(-2, -1))
    ref_norm = jnp.linalg.norm(ref, axis=(-2, -1))
    ratios = jnp.where(ref_norm > 1e-15, diff / ref_norm, diff)
    return float(jnp.mean(ratios))


def _bounded(value, low, high):
    """Clamp value to [low, high] range."""
    return max(low, min(high, value))


def compute_reference(problem: BenchmarkProblem, method: Method, tag: str):
    """Compute reference solution following the problem's method precedence.

    Uses Expm for small problems -> high-tolerance Dopri8 otherwise.
    Caches the reference on the problem instance for reuse.
    """
    if problem._reference is not None:
        return problem._reference

    ref_states = None
    ref_expects = None

    if tag == 'analytical':
        # Try Expm first (diagonalisation) for small systems
        try:
            if hasattr(problem, 'Ls'):
                ref = dq.mesolve(
                    problem.H,
                    problem.Ls,
                    problem.y0,
                    problem.tsave,
                    exp_ops=problem.Es,
                    method=Expm(),
                    progress_meter=False,
                )
            else:
                ref = dq.sesolve(
                    problem.H,
                    problem.y0,
                    problem.tsave,
                    exp_ops=problem.Es,
                    method=Expm(),
                    progress_meter=False,
                )
            ref.block_until_ready()
            ref_states = ref.states
            ref_expects = ref.expects
        except (TypeError, ValueError):
            pass

    # Fallback: high-tolerance adaptive
    if ref_states is None:
        try:
            high_order = Dopri8(rtol=1e-10, atol=1e-10)
            if hasattr(problem, 'Ls'):
                ref = dq.mesolve(
                    problem.H,
                    problem.Ls,
                    problem.y0,
                    problem.tsave,
                    exp_ops=problem.Es,
                    method=high_order,
                    progress_meter=False,
                )
            else:
                ref = dq.sesolve(
                    problem.H,
                    problem.y0,
                    problem.tsave,
                    exp_ops=problem.Es,
                    method=high_order,
                    progress_meter=False,
                )
            ref.block_until_ready()
            ref_states = ref.states
            ref_expects = ref.expects
        except (TypeError, ValueError):
            pass

    problem._reference = (ref_states, ref_expects)
    return problem._reference


def run_benchmark() -> list[BenchmarkResult]:
    runner = BenchmarkRunner()
    problems = [cls() for cls in ALL_PROBLEMS]

    for problem in problems:
        for method_name, (method_instance, ref_tag) in problem.methods.items():
            ref_states, _ = compute_reference(problem, method_instance, ref_tag)

            # JIT compile the solver call
            try:
                if hasattr(problem, 'Ls'):
                    compiled = dq.mesolve(
                        problem.H,
                        problem.Ls,
                        problem.y0,
                        problem.tsave,
                        exp_ops=problem.Es,
                        method=method_instance,
                        progress_meter=False,
                    )
                else:
                    compiled = dq.sesolve(
                        problem.H,
                        problem.y0,
                        problem.tsave,
                        exp_ops=problem.Es,
                        method=method_instance,
                        progress_meter=False,
                    )
                compiled.block_until_ready()
            except (TypeError, ValueError) as e:
                runner.results.append(
                    BenchmarkResult(
                        problem=problem.name,
                        method=method_name,
                        runtime_s=0.0,
                        nsteps=0,
                        fidelity_error=float('nan'),
                        state_error=float('nan'),
                        status=f'fail: {e!r}'[:60],
                    )
                )
                continue

            # Timed run
            try:
                t0 = time_module.perf_counter()
                if hasattr(problem, 'Ls'):
                    result = dq.mesolve(
                        problem.H,
                        problem.Ls,
                        problem.y0,
                        problem.tsave,
                        exp_ops=problem.Es,
                        method=method_instance,
                        progress_meter=False,
                    )
                else:
                    result = dq.sesolve(
                        problem.H,
                        problem.y0,
                        problem.tsave,
                        exp_ops=problem.Es,
                        method=method_instance,
                        progress_meter=False,
                    )
                result.block_until_ready()
                elapsed = time_module.perf_counter() - t0

                nsteps = int(getattr(result.infos, 'nsteps', 0))

                fidel_err = _fidelity_error(result.states, ref_states)
                l2_err = _l2_state_error(result.states, ref_states)

                runner.results.append(
                    BenchmarkResult(
                        problem=problem.name,
                        method=method_name,
                        runtime_s=elapsed,
                        nsteps=nsteps,
                        fidelity_error=fidel_err,
                        state_error=l2_err,
                        status='pass',
                    )
                )
            except (TypeError, ValueError) as e:
                runner.results.append(
                    BenchmarkResult(
                        problem=problem.name,
                        method=method_name,
                        runtime_s=0.0,
                        nsteps=0,
                        fidelity_error=float('nan'),
                        state_error=float('nan'),
                        status=f'fail: {e!r}'[:60],
                    )
                )

    runner.save_csv()
    runner.print_leaderboard()
    return runner.results


if __name__ == '__main__':
    run_benchmark()
