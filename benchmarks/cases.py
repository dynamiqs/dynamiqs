"""Benchmark case registry: a minimal set of representative user workloads."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

import dynamiqs as dq


@dataclass(frozen=True)
class Case:
    """A single benchmark case.

    Attributes:
        name: Benchmark family name, e.g. `'sesolve_kerr'`.
        params: Case parameters, e.g. `{'n': 128, 'batch': 100}`. Together with `name`
            they identify a case across runs (used to align rows in `compare`).
        build: Zero-argument setup function (not timed) returning the zero-argument
            run closure (timed). The closure must return a JAX pytree, which the
            runner blocks on with `jax.block_until_ready()`.
    """

    name: str
    params: dict[str, Any]
    build: Callable[[], Callable[[], Any]]

    @property
    def key(self) -> str:
        params = ','.join(f'{k}={v}' for k, v in sorted(self.params.items()))
        return f'{self.name}[{params}]'


def _sesolve_kerr(n: int, batch: int) -> Callable[[], Any]:
    # driven Kerr oscillator (closed system, modulated drive, optional batching)
    a = dq.destroy(n)
    H0 = -0.5 * a.dag() @ a.dag() @ a @ a
    amps = 0.5 if batch == 1 else jnp.linspace(0.1, 1.0, batch)
    H = H0 + dq.modulated(lambda t: amps * jnp.cos(2.0 * t), a + a.dag())
    psi0 = dq.coherent(n, 2.0)
    tsave = jnp.linspace(0.0, 10.0, 101)

    @eqx.filter_jit
    def run() -> dq.SESolveResult:
        return dq.sesolve(H, psi0, tsave, progress_meter=False)

    return run


def _mesolve_cavity(n: int) -> Callable[[], Any]:
    # driven-damped cavity (open system, constant Hamiltonian and jump operator)
    a = dq.destroy(n)
    H = 1.0 * dq.number(n) + 0.5 * (a + a.dag())
    Ls = [a]  # kappa = 1
    rho0 = dq.coherent_dm(n, 1.0)
    tsave = jnp.linspace(0.0, 5.0, 101)

    @eqx.filter_jit
    def run() -> dq.MESolveResult:
        return dq.mesolve(H, Ls, rho0, tsave, progress_meter=False)

    return run


def _mesolve_cat(n: int, alpha: float, batch: int) -> Callable[[], Any]:
    # cat qubit: two-photon dissipation with a batched Zeno drive
    a = dq.destroy(n)
    L2 = a @ a - alpha**2 * dq.eye(n)  # kappa_2 = 1
    eps = 0.3 if batch == 1 else jnp.linspace(0.0, 0.3, batch)[:, None, None]
    H = eps * (a + a.dag())
    rho0 = dq.fock_dm(n, 0)
    tsave = jnp.linspace(0.0, 4.0, 101)

    @eqx.filter_jit
    def run() -> dq.MESolveResult:
        return dq.mesolve(H, [L2], rho0, tsave, progress_meter=False)

    return run


def _sesolve_pwc(n: int, nseg: int) -> Callable[[], Any]:
    # piecewise-constant drive (closed system, adaptive stepper crossing the
    # segment boundaries -- `nrejected` tracks the cost of these discontinuities)
    a = dq.destroy(n)
    times = jnp.linspace(0.0, 10.0, nseg + 1)
    values = jnp.sin(jnp.pi * jnp.arange(nseg) / nseg)
    H = 1.0 * dq.number(n) + dq.pwc(times, values, a + a.dag())
    psi0 = dq.coherent(n, 1.0)
    tsave = jnp.linspace(0.0, 10.0, 101)

    @eqx.filter_jit
    def run() -> dq.SESolveResult:
        return dq.sesolve(H, psi0, tsave, progress_meter=False)

    return run


def _mesolve_grad(n: int) -> Callable[[], Any]:
    # gradient of a scalar loss through mesolve (pulse-optimization workload)
    a = dq.destroy(n)
    number_op = dq.number(n)
    rho0 = dq.coherent_dm(n, 1.0)
    tsave = jnp.linspace(0.0, 5.0, 101)

    def loss(eps: jax.Array) -> jax.Array:
        H = 1.0 * number_op + eps * (a + a.dag())
        gradient = dq.gradient.BackwardCheckpointed()
        result = dq.mesolve(
            H, [a], rho0, tsave, gradient=gradient, progress_meter=False
        )
        return dq.expect(number_op, result.final_state).real

    grad_fn = jax.jit(jax.grad(loss))
    return lambda: grad_fn(jnp.array(0.5))


def benchmark_cases(quick: bool = False) -> list[Case]:
    """Return the list of benchmark cases.

    Args:
        quick: If `True`, use tiny problem sizes (for CPU CI and sanity runs).
    """
    if quick:
        kerr = [(8, 2)]
        cavity = [8]
        cat = [(8, 1.0, 2)]
        pwc = [(8, 20)]
        grad = [8]
    else:
        kerr = [(128, 1), (128, 100), (1024, 1), (1024, 100)]
        cavity = [32, 128]
        cat = [(32, 2.0, 1), (32, 2.0, 10)]
        pwc = [(128, 100)]
        grad = [32]

    partial = functools.partial
    cases = []
    for n, batch in kerr:
        build = partial(_sesolve_kerr, n, batch)
        cases.append(Case('sesolve_kerr', {'n': n, 'batch': batch}, build))
    for n in cavity:
        cases.append(Case('mesolve_cavity', {'n': n}, partial(_mesolve_cavity, n)))
    for n, alpha, batch in cat:
        build = partial(_mesolve_cat, n, alpha, batch)
        params = {'n': n, 'alpha': alpha, 'batch': batch}
        cases.append(Case('mesolve_cat', params, build))
    for n, nseg in pwc:
        build = partial(_sesolve_pwc, n, nseg)
        cases.append(Case('sesolve_pwc', {'n': n, 'nseg': nseg}, build))
    for n in grad:
        cases.append(Case('mesolve_grad', {'n': n}, partial(_mesolve_grad, n)))
    return cases
