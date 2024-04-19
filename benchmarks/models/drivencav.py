from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np
import qutip as qt
from qutip.solver.result import Result as qtResult

import dynamiqs as dq
from dynamiqs.result import Result as dqResult


class DrivenCavity:
    """Model of a driven cavity."""

    def __init__(
        self,
        eta: float = 1.5,
        delta: float = 0.2,
        alpha0: complex = 0.3 - 0.5j,
        time: float = 10.0,
        num_tsave: int = 11,
        N: int = 128,
        qutip_data_format: Literal['dense', 'csr', 'dia'] = 'dia',
    ):
        # === prepare generic objects
        # time evolution
        tsave = np.linspace(0.0, time, num_tsave)

        # operators
        a = dq.destroy(N)

        # Hamiltonian
        H = delta * dq.dag(a) @ a + eta * (a + dq.dag(a))

        # initial state
        psi0 = dq.coherent(N, alpha0)

        # === prepare dynamiqs arguments
        self.args_dynamiqs = (H, psi0, tsave)

        # === prepare qutip arguments
        # convert arrays to qutip objects
        dims = [N]
        H = dq.to_qutip(H, dims=dims).to(qutip_data_format)
        psi0 = dq.to_qutip(psi0, dims=dims)

        # init arguments
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
