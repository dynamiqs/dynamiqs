"""Representation of a cross-resonance modulated sesolve problem to solve.

In this context, we simulate closed two-qubit Schrödinger evolution inspired
by a cross-resonance gate, with a constant interaction Hamiltonian and a
modulated drive.

Based-on: https://github.com/Nicolas-Lepage/dynamiqs/blob/feat/benchmark_MVP/tests/benchmark/bench_cross_resonance_modulated_sesolve.py
"""

import jax
import jax.numpy as jnp

import dynamiqs as dq

# Model parameters from cross-resonance.py
OMEGA_1 = 4.0
OMEGA_2 = 6.0
J = 0.4
EPSILON = 0.4

NUM_TSAVE = 100


def build_problem() -> tuple[dq.TimeQArray, dq.QArray, jax.Array]:
    """Build a simulation of cross-resonance modulated sesolve problem.

    More specifically, this problem represents a closed two-qubit Schrödinger
    evolution inspired by a cross-resonance gate, with a constant interaction
    Hamiltonian and a modulated drive.

    Returns:
        The Hamiltonian and initial state of the system of the problem, along with
        the times at which the states and expectation values should be saved.
    """
    gate_time = 0.5 * jnp.pi * jnp.abs(OMEGA_2 - OMEGA_1) / (J * EPSILON)
    tsave = jnp.linspace(0.0, gate_time, NUM_TSAVE)

    sz1 = dq.tensor(dq.sigmaz(), dq.eye(2))
    sz2 = dq.tensor(dq.eye(2), dq.sigmaz())

    sp1 = dq.tensor(dq.sigmap(), dq.eye(2))
    sp2 = dq.tensor(dq.eye(2), dq.sigmap())

    sm1 = dq.tensor(dq.sigmam(), dq.eye(2))
    sm2 = dq.tensor(dq.eye(2), dq.sigmam())

    omega_d = OMEGA_2 - J**2 / (OMEGA_1 - OMEGA_2)

    H0 = 0.5 * OMEGA_1 * sz1 + 0.5 * OMEGA_2 * sz2 + J * (sp1 @ sm2 + sm1 @ sp2)

    Hd = EPSILON * (sp1 + sm1)

    def fd(t):
        return jnp.cos(omega_d * t)

    H = H0 + dq.modulated(fd, Hd)

    psi0 = dq.tensor(dq.basis(2, 1), dq.basis(2, 1))

    return H, psi0, tsave


__all__ = ['build_problem']
