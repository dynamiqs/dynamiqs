from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import qutip as qt

import dynamiqs as dq

from .bench_model import BenchModel


class CatCNOT(BenchModel):
    """Model of a CNOT between dissipative cat qubits.

    For more details on the model, see [Guillaud, Jérémie, and Mazyar Mirrahimi.
    "Repetition cat qubits for fault-tolerant quantum computation." Physical Review X
    9.4 (2019): 041053.]
    """

    def init_qutip(
        self,
        kappa_2: float = 1.0,
        g_cnot: float = 0.3,
        nbar: float = 4.0,
        num_tsave: int = 100,
        N: int = 16,
    ):
        # time evolution
        alpha = np.sqrt(nbar)
        gate_time = np.pi / (4 * alpha * g_cnot)
        tlist = np.linspace(0.0, gate_time, num_tsave)

        # operators
        ac = qt.tensor(qt.destroy(N), qt.qeye(N))
        nt = qt.tensor(qt.qeye(N), qt.num(N))

        # Hamiltonian
        H = g_cnot * (ac + ac.dag()) * (nt - nbar)

        # collapse operators
        c_ops = [np.sqrt(kappa_2) * (ac**2 - nbar)]

        # initial state
        plus = (qt.coherent(N, alpha) + qt.coherent(N, -alpha)).unit()
        psi0 = qt.tensor(plus, plus)

        self.kwargs_qutip = {'H': H, 'rho0': psi0, 'tlist': tlist, 'c_ops': c_ops}
        self.fn_qutip = qt.mesolve

    def init_dynamiqs(
        self,
        kappa_2: float = 1.0,
        g_cnot: float = 0.3,
        nbar: float = 4.0,
        num_tsave: int = 100,
        N: int = 16,
    ):
        # time evolution
        alpha = jnp.sqrt(nbar)
        gate_time = jnp.pi / (4 * alpha * g_cnot)
        tsave = jnp.linspace(0.0, gate_time, num_tsave)

        # operators
        ac = dq.tensor(dq.destroy(N), dq.eye(N))
        nt = dq.tensor(dq.eye(N), dq.number(N))
        I = dq.tensor(dq.eye(N), dq.eye(N))

        # Hamiltonian
        H = g_cnot * (ac + dq.dag(ac)) @ (nt - nbar * I)

        # jump operator
        jump_ops = [jnp.sqrt(kappa_2) * (ac @ ac - nbar * I)]

        # initial state
        plus = dq.unit(dq.coherent(N, alpha) + dq.coherent(N, -alpha))
        psi0 = dq.tensor(plus, plus)

        self.kwargs_dynamiqs = {
            'H': H,
            'jump_ops': jump_ops,
            'rho0': psi0,
            'tsave': tsave,
        }
        self.fn_dynamiqs = dq.mesolve
