from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

import dynamiqs as dq

Kind = Literal['sesolve', 'mesolve']
Profile = Literal['smoke', 'standard', 'full']
ReferenceKind = Literal['state', 'expect']


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    kind: Kind
    description: str
    reference_strategy: str
    H: object
    jump_ops: list[object]
    y0: object
    tsave: object
    exp_ops: list[object]
    reference_kind: ReferenceKind = 'state'
    reference_expect: object | None = None
    tags: tuple[str, ...] = ()


def benchmark_cases(profile: Profile = 'standard') -> list[BenchmarkCase]:
    """Return the representative Dynamiqs solver benchmark cases.

    The smoke profile is intentionally tiny for CI. The standard/full profiles keep the
    same physics while increasing Hilbert-space sizes, number of saved times, or batch
    size so CSV files remain comparable across commits.
    """
    return [
        cross_resonance_sesolve(profile),
        driven_damped_oscillator(profile),
        batched_kerr_mesolve(profile),
        ising_chain_sesolve(profile),
        two_mode_pwc_mesolve(profile),
        reduced_cnot_mesolve(profile),
    ]


def cross_resonance_sesolve(profile: Profile) -> BenchmarkCase:
    nsave = {'smoke': 9, 'standard': 101, 'full': 201}[profile]
    tfinal = 1.0
    tsave = jnp.linspace(0.0, tfinal, nsave)
    sx, sy, sz, ident = dq.sigmax(), dq.sigmay(), dq.sigmaz(), dq.eye(2)
    ix = dq.tensor(ident, sx)
    iy = dq.tensor(ident, sy)
    zx = dq.tensor(sz, sx)
    zz = dq.tensor(sz, sz)
    drive_amp = 0.35
    drive_freq = 2.5

    def in_phase(t):
        return drive_amp * jnp.cos(2 * jnp.pi * drive_freq * t)

    def quadrature(t):
        return 0.4 * drive_amp * jnp.sin(2 * jnp.pi * drive_freq * t)

    H = (
        0.08 * zz
        + 0.05 * zx
        + dq.modulated(in_phase, ix)
        + dq.modulated(quadrature, iy)
    )
    
    return BenchmarkCase(
        name='cross_resonance_modulated_sesolve',
        kind='sesolve',
        description='Closed two-qubit cross-resonance-like gate with modulated drives.',
        reference_strategy='Dopri8, double precision, rtol=atol=1e-10, safety_factor=0.75.',
        H=H,
        jump_ops=[],
        y0=dq.fock((2, 2), (0, 0)),
        tsave=tsave,
        exp_ops=[zz, ix],
        tags=('closed', 'two-qubit', 'time-dependent'),
    )


def driven_damped_oscillator(profile: Profile) -> BenchmarkCase:
    dim = {'smoke': 5, 'standard': 16, 'full': 24}[profile]
    nsave = {'smoke': 9, 'standard': 121, 'full': 241}[profile]
    tsave = jnp.linspace(0.0, 6.0, nsave)
    kappa = 0.35
    eps = 0.22 + 0.08j
    a = dq.destroy(dim)
    H = 1j * (eps * a.dag() - jnp.conj(eps) * a)
    alpha_ref = (2.0 * eps / kappa) * (1.0 - jnp.exp(-0.5 * kappa * tsave))
    return BenchmarkCase(
        name='driven_damped_oscillator_mesolve',
        kind='mesolve',
        description='Driven-damped harmonic oscillator initialized in vacuum.',
        reference_strategy='Analytical oscillator amplitude <a>(t)=2 eps/kappa (1-exp(-kappa t/2)).',
        H=H,
        jump_ops=[jnp.sqrt(kappa) * a],
        y0=dq.fock_dm(dim, 0),
        tsave=tsave,
        exp_ops=[a],
        reference_kind='expect',
        reference_expect=alpha_ref[jnp.newaxis, :],
        tags=('open', 'one-mode', 'analytical-reference'),
    )


def batched_kerr_mesolve(profile: Profile) -> BenchmarkCase:
    dim = {'smoke': 5, 'standard': 12, 'full': 18}[profile]
    batch = {'smoke': 2, 'standard': 8, 'full': 24}[profile]
    nsave = {'smoke': 7, 'standard': 81, 'full': 161}[profile]
    tsave = jnp.linspace(0.0, 3.0, nsave)
    a = dq.destroy(dim)
    n = dq.number(dim)
    drives = jnp.linspace(0.03, 0.22, batch)
    H0 = -0.04 * n + 0.015 * n @ (n - dq.eye(dim))
    H = H0 + drives[:, None, None] * (a + a.dag())
    return BenchmarkCase(
        name='batched_kerr_oscillator_mesolve',
        kind='mesolve',
        description='Batched nonlinear Kerr oscillator with damping and drive sweep.',
        reference_strategy='Expm for smoke/standard constant Liouvillian; tight Dopri8 is a fallback for larger dimensions.',
        H=H,
        jump_ops=[jnp.sqrt(0.08) * a],
        y0=dq.coherent_dm(dim, 0.5),
        tsave=tsave,
        exp_ops=[n, a + a.dag()],
        reference_kind='expect',
        tags=('open', 'one-mode', 'batched', 'nonlinear'),
    )


def ising_chain_sesolve(profile: Profile) -> BenchmarkCase:
    nqubits = {'smoke': 3, 'standard': 8, 'full': 12}[profile]
    nsave = {'smoke': 6, 'standard': 41, 'full': 81}[profile]
    tsave = jnp.linspace(0.0, 1.5, nsave)
    sx, sz, ident = dq.sigmax(), dq.sigmaz(), dq.eye(2)

    def op_at(op, site):
        return dq.tensor(*[op if i == site else ident for i in range(nqubits)])

    H = 0
    for i in range(nqubits - 1):
        H = H + 0.12 * op_at(sz, i) @ op_at(sz, i + 1)
    for i in range(nqubits):
        H = H + 0.07 * op_at(sx, i)
    psi0 = dq.tensor(*[(dq.fock(2, 0) + dq.fock(2, 1)).unit() for _ in range(nqubits)])
    return BenchmarkCase(
        name=f'ising_chain_{nqubits}q_sesolve',
        kind='sesolve',
        description='Closed nearest-neighbor transverse-field Ising chain.',
        reference_strategy='High-accuracy Dopri8 in double precision; Expm can be enabled for smaller constant-H cases.',
        H=H,
        jump_ops=[],
        y0=psi0,
        tsave=tsave,
        exp_ops=[op_at(sz, 0), op_at(sz, nqubits - 1)],
        tags=('closed', 'many-body', 'scaling'),
    )


def two_mode_pwc_mesolve(profile: Profile) -> BenchmarkCase:
    dims = {'smoke': (3, 2), 'standard': (6, 4), 'full': (8, 6)}[profile]
    batch = {'smoke': 2, 'standard': 4, 'full': 8}[profile]
    ninterval = {'smoke': 3, 'standard': 12, 'full': 24}[profile]
    times = jnp.linspace(0.0, 4.0, ninterval + 1)
    tsave = jnp.linspace(0.0, 4.0, {'smoke': 7, 'standard': 81, 'full': 161}[profile])
    a, b = dq.destroy(*dims)
    na, nb = a.dag() @ a, b.dag() @ b
    base = -0.03 * na - 0.05 * nb + 0.01 * na @ (na - dq.eye(*dims)) + 0.02 * na @ nb
    values = 0.12 * jnp.sin(jnp.linspace(0.0, jnp.pi, ninterval))[None, :]
    values = values * jnp.linspace(0.7, 1.3, batch)[:, None]
    H = base + dq.pwc(times, values, a + a.dag()) + 0.05 * (b + b.dag())
    return BenchmarkCase(
        name='two_mode_pwc_batched_mesolve',
        kind='mesolve',
        description='Two-mode Kerr/cross-Kerr open system with batched piecewise-constant drive.',
        reference_strategy='High-accuracy Dopri8 or Rouchon3 in double precision, comparing expectation values.',
        H=H,
        jump_ops=[jnp.sqrt(0.05) * a, jnp.sqrt(0.09) * b],
        y0=dq.fock_dm(dims, (0, 0)),
        tsave=tsave,
        exp_ops=[na, nb],
        reference_kind='expect',
        tags=('open', 'two-mode', 'pwc', 'batched'),
    )


def reduced_cnot_mesolve(profile: Profile) -> BenchmarkCase:
    dims = {'smoke': (2, 2, 2), 'standard': (2, 2, 4), 'full': (2, 2, 6)}[profile]
    nsave = {'smoke': 6, 'standard': 61, 'full': 121}[profile]
    tsave = jnp.linspace(0.0, 2.5, nsave)
    c, t, buf = dq.destroy(*dims)
    nc, nt, nb = c.dag() @ c, t.dag() @ t, buf.dag() @ buf
    H = 0.04 * nc @ nt + 0.18 * nt @ nb + 0.09 * (buf + buf.dag())
    gamma = 0.45
    return BenchmarkCase(
        name='reduced_zeno_cnot_mesolve',
        kind='mesolve',
        description='Reduced three-mode dissipative gate-like model inspired by Zeno-CNOT use cases.',
        reference_strategy='High-accuracy Rouchon3/Dopri8 in double precision, validating final populations and occupations.',
        H=H,
        jump_ops=[jnp.sqrt(gamma) * buf],
        y0=dq.fock_dm(dims, (1, 0, 0)),
        tsave=tsave,
        exp_ops=[nc, nt, nb],
        reference_kind='expect',
        tags=('open', 'three-mode', 'gate-like', 'dissipative'),
    )
