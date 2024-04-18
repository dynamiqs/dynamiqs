from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np
import qutip as qt
from qutip.solver.result import Result as qtResult

import dynamiqs as dq
from dynamiqs.result import Result as dqResult


class CatCNOT:
    """Model of a CNOT between dissipative cat qubits.

    For more details on the model, see [Guillaud, Jérémie, and Mazyar Mirrahimi.
    "Repetition cat qubits for fault-tolerant quantum computation." Physical Review X
    9.4 (2019): 041053.]
    """

    def __init__(
        self,
        kappa_2: float = 1.0,
        g_cnot: float = 0.3,
        nbar: float = 4.0,
        num_tsave: int = 100,
        N: int = 16,
        qutip_data_format: Literal['dense', 'csr', 'dia'] = 'dia',
    ):
        # === prepare generic objects
        # time evolution
        alpha = jnp.sqrt(nbar)
        gate_time = jnp.pi / (4 * alpha * g_cnot)
        tsave = np.linspace(0.0, gate_time, num_tsave)

        # operators
        ac = dq.tensor(dq.destroy(N), dq.eye(N))
        at = dq.tensor(dq.eye(N), dq.destroy(N))
        i = dq.tensor(dq.eye(N), dq.eye(N))

        # Hamiltonian
        H = g_cnot * (ac + dq.dag(ac)) @ (dq.dag(at) @ at - nbar * i)

        # jump operator
        jump_ops = [jnp.sqrt(kappa_2) * (ac @ ac - nbar * i)]

        # initial state
        plus = dq.unit(dq.coherent(N, alpha) + dq.coherent(N, -alpha))
        psi0 = dq.tensor(plus, plus)

        # === prepare dynamiqs arguments
        self.args_dynamiqs = (H, jump_ops, psi0, tsave)

        # === prepare qutip arguments
        # convert arrays to qutip objects
        dims = [N, N]
        H = dq.to_qutip(H, dims=dims, data_format=qutip_data_format)
        psi0 = dq.to_qutip(psi0, dims=dims)
        c_ops = [dq.to_qutip(jump_ops[0], dims=dims, data_format=qutip_data_format)]

        # init arguments
        self.args_qutip = (H, psi0, tsave, c_ops)

    def run_dynamiqs(self) -> dqResult:
        return dq.mesolve(*self.args_dynamiqs)

    def run_qutip(self) -> qtResult:
        return qt.mesolve(*self.args_qutip)

    def check_equal(self) -> bool:
        states_dynamiqs = self.run_dynamiqs().states
        states_qutip = self.run_qutip().states
        states_qutip = jnp.stack(
            [states_qutip[i].full() for i in range(len(states_qutip))]
        )
        return jnp.allclose(states_dynamiqs, states_qutip, atol=1e-4)
