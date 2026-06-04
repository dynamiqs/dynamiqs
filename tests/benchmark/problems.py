from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp

import dynamiqs as dq
from dynamiqs.gradient import Direct
from dynamiqs.method import Dopri8, Method
from dynamiqs.progress_meter import NoProgressMeter

from .metrics import max_abs_error, relative_l2_error

ProblemKind = Literal['sesolve', 'mesolve']


@dataclass(frozen=True)
class BenchmarkProblem:
    name: str
    kind: ProblemKind
    description: str
    deterministic_methods: Sequence[str]
    reference_name: str
    run: Callable[[Method], Any]
    reference: Callable[[], Any]
    error: Callable[[Any, Any], float]


def _solver_options() -> dict[str, Any]:
    return {'save_states': False, 'progress_meter': NoProgressMeter()}


def _expect_trajectory(result: Any, exp_index: int = 0) -> jnp.ndarray:
    if result.expects is None:
        raise ValueError('benchmark result does not contain expectation values')
    return jnp.asarray(result.expects[exp_index])


def cross_resonance_modulated_sesolve() -> BenchmarkProblem:
    omega_1 = 4.0
    omega_2 = 6.0
    coupling = 0.4
    epsilon = 0.4
    nsave = 80
    gate_time = 0.5 * jnp.pi * jnp.abs(omega_2 - omega_1) / (coupling * epsilon)
    tsave = jnp.linspace(0.0, gate_time, nsave)

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
    exp_ops = [sz1, sz2]

    def run(method: Method) -> Any:
        return dq.sesolve(
            hamiltonian,
            psi0,
            tsave,
            exp_ops=exp_ops,
            method=method,
            gradient=Direct(),
            **_solver_options(),
        )

    def reference() -> jnp.ndarray:
        method = Dopri8(rtol=1e-10, atol=1e-10, safety_factor=0.75, max_steps=1_000_000)
        result = run(method)
        result.block_until_ready()
        return jnp.asarray(result.expects)

    return BenchmarkProblem(
        name='cross_resonance_modulated_sesolve',
        kind='sesolve',
        description='Closed two-qubit cross-resonance-inspired modulated evolution.',
        deterministic_methods=(
            'Tsit5',
            'Dopri5',
            'Dopri8',
            'Kvaerno3',
            'Kvaerno5',
            'Euler',
        ),
        reference_name='Dopri8(rtol=atol=1e-10,safety_factor=0.75)',
        run=run,
        reference=reference,
        error=lambda result, ref: relative_l2_error(result.expects, ref),
    )


def driven_damped_harmonic_oscillator() -> BenchmarkProblem:
    trunc = 32
    omega = 1.0
    kappa = 0.05
    epsilon = 2.0
    nsave = 80
    t_final = 2.0 * jnp.pi / omega
    tsave = jnp.linspace(0.0, t_final, nsave)

    destroy = dq.destroy(trunc)
    hamiltonian = omega * destroy.dag() @ destroy + epsilon * (destroy + destroy.dag())
    jump_ops = [jnp.sqrt(kappa) * destroy]
    rho0 = dq.fock_dm(trunc, 0)
    exp_ops = [destroy]

    def alpha_reference() -> jnp.ndarray:
        rate = kappa / 2.0 + 1j * omega
        alpha_ss = -1j * epsilon / rate
        return alpha_ss * (1.0 - jnp.exp(-rate * tsave))

    def run(method: Method) -> Any:
        return dq.mesolve(
            hamiltonian,
            jump_ops,
            rho0,
            tsave,
            exp_ops=exp_ops,
            method=method,
            gradient=Direct(),
            **_solver_options(),
        )

    return BenchmarkProblem(
        name='driven_damped_harmonic_oscillator',
        kind='mesolve',
        description='One-mode driven damped oscillator with analytical amplitude.',
        deterministic_methods=(
            'Tsit5',
            'Dopri5',
            'Dopri8',
            'Kvaerno3',
            'Kvaerno5',
            'Euler',
            'Rouchon1',
            'Rouchon2',
            'Rouchon3',
            'Expm',
        ),
        reference_name='analytical oscillator amplitude',
        run=run,
        reference=alpha_reference,
        error=lambda result, ref: relative_l2_error(_expect_trajectory(result), ref),
    )


def batched_kerr_oscillator_mesolve() -> BenchmarkProblem:
    trunc = 18
    detuning = 0.4
    kerr = 0.08
    kappa = 0.12
    drive_amplitudes = jnp.asarray([0.4, 0.7, 1.0])
    tsave = jnp.linspace(0.0, 4.0, 60)

    destroy = dq.destroy(trunc)
    number = destroy.dag() @ destroy
    h0 = (
        detuning * number
        + 0.5 * kerr * destroy.dag() @ destroy.dag() @ destroy @ destroy
    )
    hamiltonian = h0 + drive_amplitudes[:, None, None] * (destroy + destroy.dag())
    jump_ops = [jnp.sqrt(kappa) * destroy]
    rho0 = dq.coherent_dm(trunc, 0.8)
    exp_ops = [number]

    def run(method: Method) -> Any:
        return dq.mesolve(
            hamiltonian,
            jump_ops,
            rho0,
            tsave,
            exp_ops=exp_ops,
            method=method,
            gradient=Direct(),
            **_solver_options(),
        )

    def reference() -> jnp.ndarray:
        result = run(
            Dopri8(rtol=1e-9, atol=1e-9, safety_factor=0.75, max_steps=1_000_000)
        )
        result.block_until_ready()
        return jnp.asarray(result.expects)

    return BenchmarkProblem(
        name='batched_kerr_oscillator_mesolve',
        kind='mesolve',
        description='Batched nonlinear Kerr oscillator with damping.',
        deterministic_methods=(
            'Tsit5',
            'Dopri5',
            'Dopri8',
            'Kvaerno3',
            'Kvaerno5',
            'Rouchon2',
            'Rouchon3',
        ),
        reference_name='Dopri8(rtol=atol=1e-9,safety_factor=0.75)',
        run=run,
        reference=reference,
        error=lambda result, ref: relative_l2_error(result.expects, ref),
    )


def ising_chain_sesolve(num_qubits: int = 8) -> BenchmarkProblem:
    coupling = 0.2
    field = 0.7
    tsave = jnp.linspace(0.0, 2.0, 50)

    sx = [
        dq.tensor(*[dq.sigmax() if i == j else dq.eye(2) for i in range(num_qubits)])
        for j in range(num_qubits)
    ]
    sz = [
        dq.tensor(*[dq.sigmaz() if i == j else dq.eye(2) for i in range(num_qubits)])
        for j in range(num_qubits)
    ]
    hamiltonian = sum(field * x for x in sx)
    hamiltonian += sum(coupling * sz[i] @ sz[i + 1] for i in range(num_qubits - 1))
    plus = (dq.basis(2, 0) + dq.basis(2, 1)).unit()
    psi0 = dq.tensor(*[plus for _ in range(num_qubits)])
    exp_ops = [sz[0]]

    def run(method: Method) -> Any:
        return dq.sesolve(
            hamiltonian,
            psi0,
            tsave,
            exp_ops=exp_ops,
            method=method,
            gradient=Direct(),
            **_solver_options(),
        )

    def reference() -> jnp.ndarray:
        result = run(
            Dopri8(rtol=1e-9, atol=1e-9, safety_factor=0.75, max_steps=1_000_000)
        )
        result.block_until_ready()
        return jnp.asarray(result.expects)

    return BenchmarkProblem(
        name=f'ising_chain_{num_qubits}q_sesolve',
        kind='sesolve',
        description='Closed nearest-neighbor Ising chain state-vector evolution.',
        deterministic_methods=(
            'Tsit5',
            'Dopri5',
            'Dopri8',
            'Kvaerno3',
            'Kvaerno5',
            'Euler',
            'Expm',
        ),
        reference_name='Dopri8(rtol=atol=1e-9,safety_factor=0.75)',
        run=run,
        reference=reference,
        error=lambda result, ref: relative_l2_error(result.expects, ref),
    )


def two_mode_pwc_vmap_mesolve() -> BenchmarkProblem:
    dims = (5, 5)
    kerr = 0.04
    cross_kerr = 0.03
    kappa = 0.08
    tsave = jnp.linspace(0.0, 3.0, 40)
    pulse_times = jnp.asarray([0.0, 0.8, 1.6, 2.4, 3.2])
    pulse_values = jnp.asarray([[0.2, 0.5, 0.1, 0.0], [0.1, 0.2, 0.4, 0.1]])

    a, b = dq.destroy(*dims)
    na = a.dag() @ a
    nb = b.dag() @ b
    h0 = kerr * (na @ na + nb @ nb) + cross_kerr * na @ nb
    drive = dq.pwc(pulse_times, pulse_values, a + a.dag())
    hamiltonian = h0 + drive
    jump_ops = [jnp.sqrt(kappa) * a, jnp.sqrt(kappa) * b]
    rho0 = dq.coherent_dm(dims, (0.6, 0.4))
    exp_ops = [na, nb]

    def run(method: Method) -> Any:
        return dq.mesolve(
            hamiltonian,
            jump_ops,
            rho0,
            tsave,
            exp_ops=exp_ops,
            method=method,
            gradient=Direct(),
            **_solver_options(),
        )

    def reference() -> jnp.ndarray:
        result = run(
            Dopri8(rtol=1e-9, atol=1e-9, safety_factor=0.75, max_steps=1_000_000)
        )
        result.block_until_ready()
        return jnp.asarray(result.expects)

    return BenchmarkProblem(
        name='two_mode_pwc_vmap_mesolve',
        kind='mesolve',
        description='Two-mode dissipative PWC drive with batched pulse amplitudes.',
        deterministic_methods=(
            'Tsit5',
            'Dopri5',
            'Dopri8',
            'Kvaerno3',
            'Kvaerno5',
            'Rouchon2',
            'Rouchon3',
        ),
        reference_name='Dopri8(rtol=atol=1e-9,safety_factor=0.75)',
        run=run,
        reference=reference,
        error=lambda result, ref: relative_l2_error(result.expects, ref),
    )


def zeno_cnot_reduced_mesolve() -> BenchmarkProblem:
    dims = (3, 3, 3)
    kappa_buffer = 2.0
    chi = 0.15
    drive = 0.4
    tsave = jnp.linspace(0.0, 2.5, 40)

    a, b, c = dq.destroy(*dims)
    na = a.dag() @ a
    nb = b.dag() @ b
    nc = c.dag() @ c
    hamiltonian = chi * na @ nb + drive * (c + c.dag()) + 0.1 * nc
    jump_ops = [jnp.sqrt(kappa_buffer) * c]
    rho0 = dq.coherent_dm(dims, (0.3, 0.7, 0.0))
    exp_ops = [na, nb, nc]

    def run(method: Method) -> Any:
        return dq.mesolve(
            hamiltonian,
            jump_ops,
            rho0,
            tsave,
            exp_ops=exp_ops,
            method=method,
            gradient=Direct(),
            **_solver_options(),
        )

    def reference() -> jnp.ndarray:
        result = run(
            Dopri8(rtol=1e-9, atol=1e-9, safety_factor=0.75, max_steps=1_000_000)
        )
        result.block_until_ready()
        return jnp.asarray(result.expects)

    return BenchmarkProblem(
        name='zeno_cnot_reduced_mesolve',
        kind='mesolve',
        description='Reduced three-mode dissipative gate-like benchmark.',
        deterministic_methods=(
            'Tsit5',
            'Dopri5',
            'Dopri8',
            'Kvaerno3',
            'Kvaerno5',
            'Rouchon2',
            'Rouchon3',
        ),
        reference_name='Dopri8(rtol=atol=1e-9,safety_factor=0.75)',
        run=run,
        reference=reference,
        error=lambda result, ref: max_abs_error(result.expects, ref),
    )


def all_problems() -> tuple[BenchmarkProblem, ...]:
    return (
        cross_resonance_modulated_sesolve(),
        driven_damped_harmonic_oscillator(),
        batched_kerr_oscillator_mesolve(),
        ising_chain_sesolve(num_qubits=8),
        two_mode_pwc_vmap_mesolve(),
        zeno_cnot_reduced_mesolve(),
    )


def smoke_problems() -> tuple[BenchmarkProblem, ...]:
    return (driven_damped_harmonic_oscillator(),)
