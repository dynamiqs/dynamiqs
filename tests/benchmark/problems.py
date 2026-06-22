from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp

import dynamiqs as dq
from dynamiqs.gradient import Direct
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
from dynamiqs.progress_meter import NoProgressMeter

from .metrics import MetricStats, state_infidelity_stats

ProblemKind = Literal['sesolve', 'mesolve']


@dataclass(frozen=True)
class MethodSpec:
    name: str
    factory: Callable[[], Method]

    @property
    def family(self) -> str:
        return self.name.split('(', maxsplit=1)[0]


def _adaptive_method_specs() -> tuple[MethodSpec, ...]:
    return (
        MethodSpec('Dopri5', Dopri5),
        MethodSpec('Dopri8', Dopri8),
        MethodSpec('Tsit5', Tsit5),
        MethodSpec('Kvaerno3', Kvaerno3),
        MethodSpec('Kvaerno5', Kvaerno5),
    )


def _euler_method_specs() -> tuple[MethodSpec, ...]:
    return tuple(
        MethodSpec(f'Euler(dt={dt:.0e})', lambda dt=dt: Euler(dt=dt))
        for dt in (1e-2, 1e-3, 1e-4)
    )


def _rouchon_method_specs() -> tuple[MethodSpec, ...]:
    return (
        *(
            MethodSpec(f'Rouchon1(dt={dt:.0e})', lambda dt=dt: Rouchon1(dt=dt))
            for dt in (1e-2, 1e-3, 1e-4)
        ),
        MethodSpec('Rouchon2', Rouchon2),
        MethodSpec('Rouchon3', Rouchon3),
    )


SESOLVE_METHODS = (*_euler_method_specs(), *_adaptive_method_specs())
MESOLVE_METHODS = (*SESOLVE_METHODS, *_rouchon_method_specs())


class BenchmarkProblem(ABC):
    name: str
    kind: ProblemKind
    description: str
    methods: tuple[MethodSpec, ...]
    nsave = 100
    reference_name = 'Dopri8(rtol=atol=1e-9,safety_factor=0.75)'

    @abstractmethod
    def run(self, method: Method) -> Any:
        pass

    def reference(self) -> Any:
        result = self.run(
            Dopri8(rtol=1e-9, atol=1e-9, safety_factor=0.75, max_steps=1_000_000)
        )
        result.block_until_ready()
        return result.states

    def error(self, result: Any, reference: Any) -> MetricStats:
        return state_infidelity_stats(result.states, reference)

    @staticmethod
    def solver_options() -> dict[str, Any]:
        return {'save_states': True, 'progress_meter': NoProgressMeter()}


class CrossResonanceModulatedSESolve(BenchmarkProblem):
    name = 'cross_resonance_modulated_sesolve'
    kind: ProblemKind = 'sesolve'
    description = 'Closed two-qubit cross-resonance-inspired modulated evolution.'
    methods = SESOLVE_METHODS

    def run(self, method: Method) -> Any:
        omega_1 = 4.0
        omega_2 = 6.0
        coupling = 0.4
        epsilon = 0.4
        gate_time = 0.5 * jnp.pi * jnp.abs(omega_2 - omega_1) / (
            coupling * epsilon
        )
        tsave = jnp.linspace(0.0, gate_time, self.nsave)

        sz1 = dq.tensor(dq.sigmaz(), dq.eye(2))
        sz2 = dq.tensor(dq.eye(2), dq.sigmaz())
        sp1 = dq.tensor(dq.sigmap(), dq.eye(2))
        sp2 = dq.tensor(dq.eye(2), dq.sigmap())
        sm1 = dq.tensor(dq.sigmam(), dq.eye(2))
        sm2 = dq.tensor(dq.eye(2), dq.sigmam())

        omega_d = omega_2 - coupling**2 / (omega_1 - omega_2)
        h0 = 0.5 * omega_1 * sz1 + 0.5 * omega_2 * sz2
        h0 += coupling * (sp1 @ sm2 + sm1 @ sp2)
        hd = epsilon * (sp1 + sm1)
        hamiltonian = h0 + dq.modulated(lambda t: jnp.cos(omega_d * t), hd)
        psi0 = dq.tensor(dq.basis(2, 1), dq.basis(2, 1))
        return dq.sesolve(
            hamiltonian,
            psi0,
            tsave,
            method=method,
            gradient=Direct(),
            **self.solver_options(),
        )


class DrivenDampedHarmonicOscillator(BenchmarkProblem):
    name = 'driven_damped_harmonic_oscillator'
    kind: ProblemKind = 'mesolve'
    description = 'One-mode driven damped oscillator with an analytical state.'
    methods = MESOLVE_METHODS
    reference_name = 'analytical coherent-state trajectory'

    trunc = 32
    omega = 1.0
    kappa = 0.05
    epsilon = 2.0

    def _tsave(self) -> Any:
        return jnp.linspace(0.0, 2.0 * jnp.pi / self.omega, self.nsave)

    def run(self, method: Method) -> Any:
        destroy = dq.destroy(self.trunc)
        hamiltonian = self.omega * destroy.dag() @ destroy
        hamiltonian += self.epsilon * (destroy + destroy.dag())
        jump_ops = [jnp.sqrt(self.kappa) * destroy]
        return dq.mesolve(
            hamiltonian,
            jump_ops,
            dq.fock_dm(self.trunc, 0),
            self._tsave(),
            method=method,
            gradient=Direct(),
            **self.solver_options(),
        )

    def reference(self) -> Any:
        rate = self.kappa / 2.0 + 1j * self.omega
        alpha_ss = -1j * self.epsilon / rate
        alpha = alpha_ss * (1.0 - jnp.exp(-rate * self._tsave()))
        return dq.coherent_dm(self.trunc, alpha)


class BatchedKerrOscillatorMESolve(BenchmarkProblem):
    name = 'batched_kerr_oscillator_mesolve'
    kind: ProblemKind = 'mesolve'
    description = 'Batched nonlinear Kerr oscillator with damping.'
    methods = MESOLVE_METHODS

    def run(self, method: Method) -> Any:
        trunc = 18
        destroy = dq.destroy(trunc)
        number = destroy.dag() @ destroy
        h0 = 0.4 * number
        h0 += 0.5 * 0.08 * destroy.dag() @ destroy.dag() @ destroy @ destroy
        drive_amplitudes = jnp.asarray([0.4, 0.7, 1.0])
        hamiltonian = h0 + drive_amplitudes[:, None, None] * (
            destroy + destroy.dag()
        )
        return dq.mesolve(
            hamiltonian,
            [jnp.sqrt(0.12) * destroy],
            dq.coherent_dm(trunc, 0.8),
            jnp.linspace(0.0, 4.0, self.nsave),
            method=method,
            gradient=Direct(),
            **self.solver_options(),
        )


class IsingChainSESolve(BenchmarkProblem):
    kind: ProblemKind = 'sesolve'
    description = 'Closed nearest-neighbor Ising chain state-vector evolution.'
    methods = (*SESOLVE_METHODS, MethodSpec('Expm', Expm))

    def __init__(self, num_qubits: int = 8):
        self.num_qubits = num_qubits
        self.name = f'ising_chain_{num_qubits}q_sesolve'

    def run(self, method: Method) -> Any:
        sx = [
            dq.tensor(
                *[
                    dq.sigmax() if i == j else dq.eye(2)
                    for i in range(self.num_qubits)
                ]
            )
            for j in range(self.num_qubits)
        ]
        sz = [
            dq.tensor(
                *[
                    dq.sigmaz() if i == j else dq.eye(2)
                    for i in range(self.num_qubits)
                ]
            )
            for j in range(self.num_qubits)
        ]
        hamiltonian = sum(0.7 * x for x in sx)
        hamiltonian += sum(
            0.2 * sz[i] @ sz[i + 1] for i in range(self.num_qubits - 1)
        )
        plus = (dq.basis(2, 0) + dq.basis(2, 1)).unit()
        psi0 = dq.tensor(*[plus for _ in range(self.num_qubits)])
        return dq.sesolve(
            hamiltonian,
            psi0,
            jnp.linspace(0.0, 2.0, self.nsave),
            method=method,
            gradient=Direct(),
            **self.solver_options(),
        )


class TwoModePWCMESolve(BenchmarkProblem):
    name = 'two_mode_pwc_vmap_mesolve'
    kind: ProblemKind = 'mesolve'
    description = 'Two-mode dissipative PWC drive with batched pulse amplitudes.'
    methods = MESOLVE_METHODS

    def run(self, method: Method) -> Any:
        dims = (5, 5)
        a, b = dq.destroy(*dims)
        na = a.dag() @ a
        nb = b.dag() @ b
        h0 = 0.04 * (na @ na + nb @ nb) + 0.03 * na @ nb
        pulse_times = jnp.asarray([0.0, 0.8, 1.6, 2.4, 3.2])
        pulse_values = jnp.asarray(
            [[0.2, 0.5, 0.1, 0.0], [0.1, 0.2, 0.4, 0.1]]
        )
        hamiltonian = h0 + dq.pwc(pulse_times, pulse_values, a + a.dag())
        return dq.mesolve(
            hamiltonian,
            [jnp.sqrt(0.08) * a, jnp.sqrt(0.08) * b],
            dq.coherent_dm(dims, (0.6, 0.4)),
            jnp.linspace(0.0, 3.0, self.nsave),
            method=method,
            gradient=Direct(),
            **self.solver_options(),
        )


class ZenoCNOTReducedMESolve(BenchmarkProblem):
    name = 'zeno_cnot_reduced_mesolve'
    kind: ProblemKind = 'mesolve'
    description = 'Reduced three-mode dissipative gate-like benchmark.'
    methods = MESOLVE_METHODS

    def run(self, method: Method) -> Any:
        dims = (3, 3, 3)
        a, b, c = dq.destroy(*dims)
        na = a.dag() @ a
        nb = b.dag() @ b
        nc = c.dag() @ c
        hamiltonian = 0.15 * na @ nb + 0.4 * (c + c.dag()) + 0.1 * nc
        return dq.mesolve(
            hamiltonian,
            [jnp.sqrt(2.0) * c],
            dq.coherent_dm(dims, (0.3, 0.7, 0.0)),
            jnp.linspace(0.0, 2.5, self.nsave),
            method=method,
            gradient=Direct(),
            **self.solver_options(),
        )


def all_problems() -> tuple[BenchmarkProblem, ...]:
    return (
        CrossResonanceModulatedSESolve(),
        DrivenDampedHarmonicOscillator(),
        BatchedKerrOscillatorMESolve(),
        IsingChainSESolve(num_qubits=8),
        TwoModePWCMESolve(),
        ZenoCNOTReducedMESolve(),
    )


def smoke_problems() -> tuple[BenchmarkProblem, ...]:
    return (DrivenDampedHarmonicOscillator(),)
