from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import qutip as qt
from qutip.solver.result import Result as qtResult

import dynamiqs as dq
from dynamiqs.result import Result as dqResult


class CrossResonanceGate:
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

    def __init__(
        self,
        omega_1: float = 4.0,
        omega_2: float = 6.0,
        J: float = 0.4,
        eps: float = 0.4,
        num_tsave: int = 100,
    ):
        # === prepare generic objects
        # time evolution
        gate_time = 0.5 * jnp.pi * abs(omega_2 - omega_1) / (J * eps)
        tsave = np.linspace(0.0, gate_time, num_tsave)

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

        # initial state
        psi0 = dq.tensor(dq.basis(2, 1), dq.basis(2, 1))

        # === prepare dynamiqs arguments
        fd = lambda t: jnp.cos(omega_d * t)
        H = H0 + dq.modulated(fd, Hd)
        self.args_dynamiqs = (H, psi0, tsave)

        # === prepare qutip arguments
        dims = [2, 2]
        H0 = dq.to_qutip(H0, dims=dims)
        Hd = dq.to_qutip(Hd, dims=dims)
        fd = lambda t: np.cos(omega_d * t)
        H = [H0, [Hd, fd]]
        psi0 = dq.to_qutip(psi0, dims=dims)
        self.args_qutip = (H, psi0, tsave)

    def run_dynamiqs(self) -> dqResult:
        return dq.sesolve(*self.args_dynamiqs)

    def run_qutip(self) -> qtResult:
        return qt.sesolve(*self.args_qutip)

    def check_equal(self) -> bool:
        states_dynamiqs = self.run_dynamiqs().states
        states_qutip = self.run_qutip().states
        states_qutip = jnp.stack(
            [states_qutip[i].full() for i in range(len(states_qutip))]
        )
        return jnp.allclose(states_dynamiqs, states_qutip, atol=1e-4)
