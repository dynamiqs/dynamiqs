from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import qutip as qt

import dynamiqs as dq

from .bench_model import BenchModel


class DrivenCavity(BenchModel):
    """Model of a driven cavity."""

    def init_qutip(
        self,
        eta: float = 1.5,
        delta: float = 0.2,
        alpha0: complex = 0.3 - 0.5j,
        time: float = 10.0,
        num_tsave: int = 10,
        N: int = 128,
    ):
        # time evolution
        tlist = np.linspace(0.0, time, num_tsave)

        # operators
        a = qt.destroy(N)

        # Hamiltonian
        H = delta * a.dag() * a + eta * (a + a.dag())

        # initial state
        psi0 = qt.coherent(N, alpha0)

        self.kwargs_qutip = {'H': H, 'psi0': psi0, 'tlist': tlist}
        self.fn_qutip = qt.sesolve

    def init_dynamiqs(
        self,
        eta: float = 1.5,
        delta: float = 0.2,
        alpha0: complex = 0.3 - 0.5j,
        time: float = 10.0,
        num_tsave: int = 10,
        N: int = 128,
    ):
        # time evolution
        tsave = jnp.linspace(0.0, time, num_tsave)

        # operators
        a = dq.destroy(N)

        # Hamiltonian
        H = delta * dq.dag(a) @ a + eta * (a + dq.dag(a))

        # initial state
        psi0 = dq.coherent(N, alpha0)

        self.kwargs_dynamiqs = {'H': H, 'psi0': psi0, 'tsave': tsave}
        self.fn_dynamiqs = dq.sesolve
