from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import qutip as qt

import dynamiqs as dq

from .bench_model import BenchModel


class CrossResonanceGate(BenchModel):
    """Model of a cross-resonance gate between two qubits, in the lab frame.

    For more details on the model, see [Magesan, Easwar, and Jay M. Gambetta.
    "Effective Hamiltonian models of the cross-resonance gate." Physical Review A 101.5
    (2020): 052308.]

    Warning:
        For this model, QuTiP 5.0.1 runtimes are very dependent on `num_tsave`. We
        essentially find a linear scaling of the runtime with `num_tsave`, suggesting
        back-and-forth communication between the QuTiP solver and Python kernel at
        every save.
    """

    def init_qutip(
        self,
        omega_1: float = 4.0,
        omega_2: float = 6.0,
        J: float = 0.4,
        eps: float = 0.4,
        num_tsave: int = 100,
    ):
        # time evolution
        gate_time = 0.5 * np.pi * abs(omega_2 - omega_1) / (J * eps)
        tlist = np.linspace(0.0, gate_time, num_tsave)

        # operators
        sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2))
        sz2 = qt.tensor(qt.qeye(2), qt.sigmaz())
        sp1 = qt.tensor(qt.sigmap(), qt.qeye(2))
        sp2 = qt.tensor(qt.qeye(2), qt.sigmap())
        sm1 = qt.tensor(qt.sigmam(), qt.qeye(2))
        sm2 = qt.tensor(qt.qeye(2), qt.sigmam())

        # Hamiltonian
        omega_d = omega_2 - J**2 / (omega_1 - omega_2)
        H0 = 0.5 * omega_1 * sz1 + 0.5 * omega_2 * sz2 + J * (sp1 * sm2 + sm1 * sp2)
        Hd = eps * (sp1 + sm1)
        fd = lambda t: np.cos(omega_d * t)
        H = [H0, [Hd, fd]]

        # initial state
        psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 1))

        self.kwargs_qutip = {'H': H, 'psi0': psi0, 'tlist': tlist}
        self.fn_qutip = qt.sesolve

    def init_dynamiqs(
        self,
        omega_1: float = 4.0,
        omega_2: float = 6.0,
        J: float = 0.4,
        eps: float = 0.4,
        num_tsave: int = 100,
    ):
        # time evolution
        gate_time = 0.5 * jnp.pi * abs(omega_2 - omega_1) / (J * eps)
        tsave = jnp.linspace(0.0, gate_time, num_tsave)

        # operators
        sz1 = dq.tensor(dq.sigmaz(), dq.eye(2))
        sz2 = dq.tensor(dq.eye(2), dq.sigmaz())
        sp1 = dq.tensor(dq.sigmap(), dq.eye(2))
        sp2 = dq.tensor(dq.eye(2), dq.sigmap())
        sm1 = dq.tensor(dq.sigmam(), dq.eye(2))
        sm2 = dq.tensor(dq.eye(2), dq.sigmam())

        # Hamiltonian
        omega_d = omega_2 - J**2 / (omega_1 - omega_2)
        H0 = 0.5 * omega_1 * sz1 + 0.5 * omega_2 * sz2 + J * (sp1 @ sm2 + sm1 @ sp2)
        Hd = eps * (sp1 + sm1)
        fd = lambda t: jnp.cos(omega_d * t)
        H = H0 + dq.modulated(fd, Hd)

        # initial state
        psi0 = dq.tensor(dq.basis(2, 1), dq.basis(2, 1))

        self.kwargs_dynamiqs = {'H': H, 'psi0': psi0, 'tsave': tsave}
        self.fn_dynamiqs = dq.sesolve
