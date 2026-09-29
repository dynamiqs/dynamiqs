"""Physics systems used by the benchmark cases. Each builder returns the ingredients
of a simulation (Hamiltonian, jump operators, initial state, save times) without
calling a solver.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

import dynamiqs as dq
from dynamiqs.qarrays.layout import Layout
from dynamiqs.qarrays.qarray import QArray
from dynamiqs.time_qarray import TimeQArray

# closed systems: (H, psi0, tsave)
ClosedSystem = tuple[QArray | TimeQArray, QArray, Array]
# open systems: (H, Ls, rho0, tsave)
OpenSystem = tuple[QArray | TimeQArray, list[QArray], QArray, Array]


def cavity(n: int, *, layout: Layout = dq.dia, batch: int = 1) -> OpenSystem:
    """Driven-damped cavity.

    Constant Hamiltonian and jump operator, both linear in `a`: the spectrum grows as
    `n`, not `n^2`, so this system stays well-conditioned as `n` grows and is the one to
    use for large-dimension sweeps.
    """
    a = dq.destroy(n, layout=layout)
    eps = 0.5 if batch == 1 else jnp.linspace(0.1, 1.0, batch)[:, None, None]
    H = 1.0 * a.dag() @ a + eps * (a + a.dag())
    Ls = [jnp.sqrt(1.0) * a]
    rho0 = dq.coherent_dm(n, 1.0)
    tsave = jnp.linspace(0.0, 5.0, 101)
    return H, Ls, rho0, tsave


def cavity_closed(n: int, *, layout: Layout = dq.dia, batch: int = 1) -> ClosedSystem:
    """Driven cavity without dissipation, the closed-system counterpart of `cavity`."""
    H, _, _, tsave = cavity(n, layout=layout, batch=batch)
    return H, dq.coherent(n, 1.0), tsave


def cat(n: int, alpha: float, *, layout: Layout = dq.dia, batch: int = 1) -> OpenSystem:
    """Cat qubit stabilized by two-photon dissipation, with a batched Zeno drive.

    The `batch` axis sweeps the drive amplitude, the shape of a parameter scan.
    """
    a = dq.destroy(n, layout=layout)
    eps = 0.3 if batch == 1 else jnp.linspace(0.0, 0.3, batch)[:, None, None]
    H = eps * (a + a.dag())
    Ls = [jnp.sqrt(1.0) * (a @ a - alpha**2 * dq.eye(n, layout=layout))]
    rho0 = dq.fock_dm(n, 0)
    tsave = jnp.linspace(0.0, 4.0, 101)
    return H, Ls, rho0, tsave


def transmon(n: int, *, layout: Layout = dq.dia) -> ClosedSystem:
    """Weakly anharmonic transmon driven by a smooth DRAG-like pulse.

    A small Hilbert space with a time-dependent Hamiltonian re-evaluated at every step:
    the regime where per-step overhead, not matrix multiplication, dominates.
    """
    a = dq.destroy(n, layout=layout)
    anharmonicity = -0.2
    H0 = 0.5 * anharmonicity * a.dag() @ a.dag() @ a @ a

    # gaussian envelope with its derivative on the quadrature (DRAG)
    tgate, sigma, amp = 20.0, 5.0, 0.05

    def envelope(t: Array) -> Array:
        return amp * jnp.exp(-0.5 * ((t - 0.5 * tgate) / sigma) ** 2)

    def drag(t: Array) -> Array:
        return -(t - 0.5 * tgate) / sigma**2 * envelope(t) / anharmonicity

    H = (
        H0
        + dq.modulated(envelope, a + a.dag())
        + dq.modulated(drag, 1j * (a - a.dag()))
    )
    return H, dq.fock(n, 0), jnp.linspace(0.0, tgate, 101)


def cross_resonance(
    n: int = 3, *, layout: Layout = dq.dia, batch: int = 1
) -> OpenSystem:
    """Cross-resonance gate between two coupled transmons, with decay and dephasing.

    Two `n`-level transmons exchange-coupled, the control driven at the target
    frequency, batched over the drive amplitude: a gate-calibration sweep in a small
    Hilbert space.
    """
    a, b = dq.destroy(n, n, layout=layout)
    wc, wt, alpha, g = 0.0, 0.3, -0.2, 0.02
    H0 = (
        wc * a.dag() @ a
        + wt * b.dag() @ b
        + 0.5 * alpha * (a.dag() @ a.dag() @ a @ a + b.dag() @ b.dag() @ b @ b)
        + g * (a.dag() @ b + a @ b.dag())
    )
    amps = 0.1 if batch == 1 else jnp.linspace(0.02, 0.2, batch)[:, None, None]
    # drive the control qubit at the target frequency
    drive = dq.modulated(lambda t: amps * jnp.cos(wt * t), a + a.dag())

    kappa, gamma = 0.005, 0.005
    Ls = [
        jnp.sqrt(kappa) * a,
        jnp.sqrt(kappa) * b,
        jnp.sqrt(gamma) * a.dag() @ a,
        jnp.sqrt(gamma) * b.dag() @ b,
    ]
    rho0 = dq.fock_dm((n, n), (1, 0))
    tsave = jnp.linspace(0.0, 50.0, 101)
    return H0 + drive, Ls, rho0, tsave


def spin_chain(nspin: int, *, layout: Layout = dq.dia) -> ClosedSystem:
    """Transverse-field Ising chain of `nspin` spins (dimension `2^nspin`).

    A many-body Hamiltonian assembled from tensor products of Pauli matrices, quenched
    from the fully polarized state: large kets, and an operator structure quite unlike
    the banded bosonic ones.
    """
    sx, sz = dq.sigmax(layout=layout), dq.sigmaz(layout=layout)
    eye = dq.eye(2, layout=layout)

    def onsite(op: QArray, i: int) -> QArray:
        return dq.tensor(*[op if j == i else eye for j in range(nspin)])

    J, h = 1.0, 0.5
    H = -J * sum(onsite(sz, i) @ onsite(sz, i + 1) for i in range(nspin - 1))
    H = H - h * sum(onsite(sx, i) for i in range(nspin))

    # all spins up; `H` carries the tensor-product dims, so `psi0` must match them
    psi0 = dq.fock((2,) * nspin, (0,) * nspin)
    tsave = jnp.linspace(0.0, 5.0, 101)
    return H, psi0, tsave


def pwc_drive(n: int, nseg: int, *, layout: Layout = dq.dia) -> ClosedSystem:
    """Cavity driven by a piecewise-constant pulse.

    The adaptive stepper must cross `nseg` discontinuities; `nrej` reports what they
    cost in rejected steps.
    """
    a = dq.destroy(n, layout=layout)
    times = jnp.linspace(0.0, 10.0, nseg + 1)
    values = jnp.sin(jnp.pi * jnp.arange(nseg) / nseg)
    H = 1.0 * dq.number(n, layout=layout) + dq.pwc(times, values, a + a.dag())
    return H, dq.coherent(n, 1.0), jnp.linspace(0.0, 10.0, 101)


def driven_kerr(n: int, *, layout: Layout = dq.dia) -> tuple[TimeQArray, float, Array]:
    """Periodically driven Kerr resonator, returned as `(H, period, tsave)`.

    The Floquet system: a strong periodic drive on a Kerr oscillator. Kept at moderate
    `n` because the Kerr spectrum grows as `n^2`, which bounds the explicit-RK step size
    by stability rather than accuracy.
    """
    a = dq.destroy(n, layout=layout)
    H0 = -0.5 * a.dag() @ a.dag() @ a @ a
    omega = 2.0
    H = H0 + dq.modulated(lambda t: 1.0 * jnp.cos(omega * t), a + a.dag())
    period = 2 * jnp.pi / omega
    tsave = jnp.linspace(0.0, period, 11)
    return H, period, tsave
