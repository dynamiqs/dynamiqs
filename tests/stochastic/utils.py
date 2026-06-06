import jax.numpy as jnp

import dynamiqs as dq

# fixed step shared by all fixed-step stochastic methods
DT = 1e-3


# ── solver runners ───────────────────────────────────────────────────────────
# Give the four stochastic solvers a common signature
#   run(H, jump_ops, psi0, tsave, keys, exp_ops) -> result
# so a single physical problem can be parametrized over all of them. The SME
# solvers use perfect efficiency (eta=1) and no dark counts (theta=0), for which
# their trajectory-averaged dynamics matches the corresponding SSE.


def run_jsse(H, jump_ops, psi0, tsave, keys, exp_ops):
    method = dq.method.EulerJump(dt=DT)
    return dq.jssesolve(
        H, jump_ops, psi0, tsave, keys=keys, exp_ops=exp_ops, method=method
    )


def run_dsse(H, jump_ops, psi0, tsave, keys, exp_ops):
    method = dq.method.EulerMaruyama(dt=DT)
    return dq.dssesolve(
        H, jump_ops, psi0, tsave, keys=keys, exp_ops=exp_ops, method=method
    )


def run_jsme(H, jump_ops, psi0, tsave, keys, exp_ops):
    thetas = jnp.zeros(len(jump_ops))
    etas = jnp.ones(len(jump_ops))
    method = dq.method.EulerJump(dt=DT)
    return dq.jsmesolve(
        H,
        jump_ops,
        thetas,
        etas,
        psi0,
        tsave,
        keys=keys,
        exp_ops=exp_ops,
        method=method,
    )


def run_dsme(H, jump_ops, psi0, tsave, keys, exp_ops):
    etas = jnp.ones(len(jump_ops))
    method = dq.method.EulerMaruyama(dt=DT)
    return dq.dsmesolve(
        H, jump_ops, etas, psi0, tsave, keys=keys, exp_ops=exp_ops, method=method
    )


SOLVERS = {'jsse': run_jsse, 'dsse': run_dsse, 'jsme': run_jsme, 'dsme': run_dsme}
JUMP_SOLVERS = ['jsse', 'jsme']


# ── physical systems ─────────────────────────────────────────────────────────


def protected_subspace_system(omega=1.0):
    # two qubits; the odd-parity subspace {|01>, |10>} is protected from the
    # measurement of L = -sz⊗sz, which acts as the identity on it
    sx, sy, sz = dq.sigmax(), dq.sigmay(), dq.sigmaz()
    H = 0.5 * omega * ((sx & sx) + (sy & sy))
    L = -(sz & sz)
    psi0 = dq.fock(2, 0) & dq.fock(2, 1)  # |01>
    return H, [L], psi0


def protected_subspace_state(t, omega=1.0):
    # exact deterministic trajectory |psi(t)> = cos(wt)|01> - i sin(wt)|10>
    ket01 = dq.fock(2, 0) & dq.fock(2, 1)
    ket10 = dq.fock(2, 1) & dq.fock(2, 0)
    return jnp.cos(omega * t) * ket01 - 1j * jnp.sin(omega * t) * ket10


def backaction_system(omega=1.0):
    # same H and initial state as `protected_subspace_system`, but with a loss
    # operator L = sz⊗I that is NOT proportional to the identity on the subspace,
    # so the measurement does introduce genuine back-action (negative control)
    sx, sy, sz = dq.sigmax(), dq.sigmay(), dq.sigmaz()
    H = 0.5 * omega * ((sx & sx) + (sy & sy))
    L = sz & dq.eye(2)
    psi0 = dq.fock(2, 0) & dq.fock(2, 1)  # |01>
    return H, [L], psi0


def decay_system(gamma=1.0):
    # spontaneously decaying qubit, H = 0, L = sqrt(gamma) sm
    H = dq.zeros(2)
    L = jnp.sqrt(gamma) * dq.sigmam()
    psi0 = dq.excited()
    return H, [L], psi0


def qnd_system(gamma=1.0):
    # qubit under QND measurement of sz, H = 0, L = sqrt(gamma) sz; psi0 is a sz
    # eigenstate, hence a fixed point with no measurement back-action
    H = dq.zeros(2)
    L = jnp.sqrt(gamma) * dq.sigmaz()
    psi0 = dq.excited()
    return H, [L], psi0


# ── helpers ──────────────────────────────────────────────────────────────────


def infidelity_with_state(result, exact_states):
    # infidelity with a pure target, for every trajectory and saved time;
    # normalized by the state norm/trace to be robust to Euler norm drift
    states = result.states.to_jax()  # (ntrajs, ntsave, n, k), k=1 (ket) or n (dm)
    exact = exact_states.to_jax()  # (ntsave, n, 1)
    bra = jnp.conj(jnp.swapaxes(exact, -1, -2))  # (ntsave, 1, n)
    if states.shape[-1] == 1:  # ket (SSE)
        overlap = (bra @ states)[..., 0, 0]  # (ntrajs, ntsave)
        bra_psi = jnp.conj(jnp.swapaxes(states, -1, -2))
        norm2 = (bra_psi @ states)[..., 0, 0].real
        return 1 - jnp.abs(overlap) ** 2 / norm2
    # density matrix (SME)
    num = (bra @ states @ exact)[..., 0, 0].real  # (ntrajs, ntsave)
    tr = jnp.trace(states, axis1=-2, axis2=-1).real
    return 1 - num / tr
