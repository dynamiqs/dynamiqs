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

import jax.numpy as jnp

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


class BenchmarkProblem:
    """Base class for a benchmark problem.

    Each subclass must define:
    - name, description
    - methods: a dict mapping method name -> (method_instance, reference_precedence)
      where reference_precedence is 'analytical' (gold standard), 'expm' (diagonalisation),
      or 'high_order' (high-tolerance adaptive).
    - run(method) that returns the solver result.
    """

    name: str
    description: str
    methods: dict[str, tuple[Method, str]]

    def run(self, method: Method) -> Any:
        raise NotImplementedError


class ClosedTwoQubit(BenchmarkProblem):
    name = 'closed_two_qubit'
    description = 'Closed two-qubit Schrödinger dynamics (cross-resonance inspired)'
    methods = {
        'Tsit5': (Tsit5(), 'analytical'),
        'Dopri5': (Dopri5(), 'analytical'),
        'Dopri8': (Dopri8(), 'analytical'),
        'Kvaerno3': (Kvaerno3(), 'analytical'),
        'Kvaerno5': (Kvaerno5(), 'analytical'),
        'Euler': (Euler(dt=1e-4), 'analytical'),
        'Rouchon1': (Rouchon1(dt=1e-4), 'analytical'),
        'Rouchon2': (Rouchon2(dt=1e-4), 'analytical'),
        'Expm': (Expm(), 'expm'),
    }

    def __init__(self):
        self.tsave = jnp.linspace(0.0, 0.3, 7)
        delta = 1.0
        eps = 0.5
        g = 0.3
        H1 = delta * dq.sigmaz()
        H2 = eps * dq.sigmax()
        H12 = g * dq.tensor(dq.sigmax(), dq.sigmax())
        self.H = dq.tensor(H1, dq.eye(2)) + dq.tensor(dq.eye(2), H2) + H12
        self.y0 = dq.tensor(dq.basis(2, 0), dq.basis(2, 1))
        self.Es = [
            dq.tensor(dq.sigmaz(), dq.eye(2)),
            dq.tensor(dq.eye(2), dq.sigmaz()),
        ]
        self._reference = None

    def run(self, method: Method) -> Any:
        return dq.sesolve(self.H, self.y0, self.tsave, exp_ops=self.Es, method=method)


class DrivenDampedHarmonicOscillator(BenchmarkProblem):
    name = 'driven_damped_oscillator'
    description = 'Driven-damped harmonic oscillator Lindblad dynamics'
    methods = {
        'Tsit5': (Tsit5(), 'analytical'),
        'Dopri5': (Dopri5(), 'analytical'),
        'Dopri8': (Dopri8(), 'analytical'),
        'Kvaerno3': (Kvaerno3(), 'analytical'),
        'Kvaerno5': (Kvaerno5(), 'analytical'),
        'Euler': (Euler(dt=1e-4), 'analytical'),
        'Rouchon1': (Rouchon1(dt=1e-4), 'analytical'),
        'Rouchon2': (Rouchon2(dt=1e-4), 'analytical'),
        'Expm': (Expm(), 'expm'),
    }

    def __init__(self):
        self.tsave = jnp.linspace(0.0, 1.2, 9)
        self.n = 6
        self.delta = 1.0
        self.kappa = 0.5
        self.y0 = dq.coherent(self.n, 1.5)
        self.H = self.delta * dq.number(self.n)
        self.Ls = [jnp.sqrt(self.kappa) * dq.destroy(self.n)]
        self.Es = [dq.position(self.n), dq.momentum(self.n)]
        self._reference = None

    def run(self, method: Method) -> Any:
        return dq.mesolve(
            self.H, self.Ls, self.y0, self.tsave, exp_ops=self.Es, method=method
        )


class BatchedKerrOscillator(BenchmarkProblem):
    name = 'batched_kerr_oscillator'
    description = 'Batched Kerr oscillator mesolve (min/max/avg stats recorded)'
    methods = {
        'Tsit5': (Tsit5(), 'analytical'),
        'Dopri5': (Dopri5(), 'analytical'),
        'Dopri8': (Dopri8(), 'analytical'),
        'Kvaerno3': (Kvaerno3(), 'analytical'),
        'Euler': (Euler(dt=1e-4), 'analytical'),
        'Rouchon2': (Rouchon2(dt=1e-4), 'analytical'),
        'Expm': (Expm(), 'expm'),
    }

    def __init__(self):
        self.tsave = jnp.linspace(0.0, 0.6, 7)
        self.n = 4
        self.omegas = jnp.linspace(0.8, 1.2, 2)
        self.y0 = dq.coherent(self.n, 1.0)
        self.H = dq.number(self.n) + 0.5 * self.omegas[..., None, None] * dq.number(self.n).elpow(2)
        self.Ls = [jnp.sqrt(0.3) * dq.destroy(self.n)]
        self.Es = [dq.number(self.n)]
        self._reference = None

    def run(self, method: Method) -> Any:
        return dq.mesolve(
            self.H, self.Ls, self.y0, self.tsave, exp_ops=self.Es, method=method
        )


class LargeScaleClosed(BenchmarkProblem):
    name = 'large_scale_closed'
    description = 'Large-scale (12-qubit) closed-system Schrödinger dynamics'
    methods = {
        'Tsit5': (Tsit5(), 'high_order'),
        'Dopri5': (Dopri5(), 'high_order'),
        'Dopri8': (Dopri8(rtol=1e-8, atol=1e-8), 'high_order'),
    }

    def __init__(self):
        self.tsave = jnp.linspace(0.0, 0.3, 7)
        n = 6
        self.H = dq.number(n) + 0.5 * dq.number(n).elpow(2)
        self.y0 = dq.coherent(n, 1.5)
        self.Es = [dq.destroy(n), dq.number(n)]
        self._reference = None

    def run(self, method: Method) -> Any:
        return dq.sesolve(self.H, self.y0, self.tsave, exp_ops=self.Es, method=method)


class TimeDependentQubit(BenchmarkProblem):
    name = 'time_dependent_qubit'
    description = 'Time-dependent qubit Schrödinger dynamics'
    methods = {
        'Tsit5': (Tsit5(), 'high_order'),
        'Dopri5': (Dopri5(), 'high_order'),
        'Dopri8': (Dopri8(rtol=1e-8, atol=1e-8), 'high_order'),
    }

    def __init__(self):
        self.tsave = jnp.linspace(0.0, 1.0, 9)
        self.eps = 2.0
        self.omega = 5.0
        self.y0 = dq.fock(2, 0)
        self.H = dq.timecallable(lambda t: self.eps * jnp.cos(self.omega * t) * dq.sigmax())
        self.Es = [dq.sigmax(), dq.sigmay(), dq.sigmaz()]
        self._reference = None

    def run(self, method: Method) -> Any:
        return dq.sesolve(self.H, self.y0, self.tsave, exp_ops=self.Es, method=method)


class OpenTimeDependentQubit(BenchmarkProblem):
    name = 'open_time_dependent_qubit'
    description = 'Open time-dependent qubit Lindblad dynamics'
    methods = {
        'Tsit5': (Tsit5(), 'high_order'),
        'Dopri5': (Dopri5(), 'high_order'),
        'Dopri8': (Dopri8(rtol=1e-8, atol=1e-8), 'high_order'),
    }

    def __init__(self):
        self.tsave = jnp.linspace(0.0, 1.0, 9)
        self.eps = 2.0
        self.omega = 5.0
        self.gamma = 0.5
        self.y0 = dq.fock(2, 0)
        self.H = dq.timecallable(lambda t: self.eps * jnp.cos(self.omega * t) * dq.sigmax())
        self.Ls = [jnp.sqrt(self.gamma) * dq.sigmax()]
        self.Es = [dq.sigmax(), dq.sigmay(), dq.sigmaz()]
        self._reference = None

    def run(self, method: Method) -> Any:
        return dq.mesolve(
            self.H, self.Ls, self.y0, self.tsave, exp_ops=self.Es, method=method
        )


# Registry of all benchmark problems
ALL_PROBLEMS: list[type[BenchmarkProblem]] = [
    ClosedTwoQubit,
    DrivenDampedHarmonicOscillator,
    BatchedKerrOscillator,
    LargeScaleClosed,
    TimeDependentQubit,
    OpenTimeDependentQubit,
]
