from __future__ import annotations

from math import pi, sqrt

from jax import numpy as jnp
import dynamiqs as dq

# units
MHz = 2 * pi
ns = 1.0 * 1e-3


class CatCNOT(OpenSystem):
    def __init__(
        self,
        N: int = 32,
        num_tslots: int = 100,
        alpha: float = 2.0,
        kappa2: float = 1.0 * MHz,
        T: float = 200 * ns,
    ):
        self.N = N
        self.num_tslots = num_tslots
        self.kappa2 = kappa2
        self.T = T

        # cnot drive amplitude
        self.g = pi / (4 * alpha * T)

        # Hamiltonian
        ac = dq.tensor(dq.destroy(N), dq.eye(N))
        at = dq.tensor(dq.eye(N), dq.destroy(N))
        i = dq.tensor(dq.eye(N), dq.eye(N))
        self.H = self.g * (ac + dq.dag(ac)) @ (dq.dag(at) @ at - alpha**2 * i)

        # jump operator
        self.jump_ops = [sqrt(kappa2) * (ac @ ac - alpha**2 * i)]

        # initial state
        plus = dq.unit(dq.coherent(N, alpha) + dq.coherent(N, -alpha))
        self.y0 = dq.tensor(plus, plus)

        # tsave
        self.tsave = jnp.linspace(0, self.T, self.num_tslots + 1)
