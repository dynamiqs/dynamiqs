from __future__ import annotations

from abc import abstractmethod

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import PyTree

import dynamiqs as dq
from dynamiqs import QArray
from dynamiqs.method import Method
from dynamiqs.result import Result
from dynamiqs.time_qarray import TimeQArray

from ._system import System


class StochasticSystem(System):
    """Base class for systems run by the four stochastic solvers.

    `run` dispatches to `jssesolve`/`dssesolve`/`jsmesolve`/`dsmesolve` given a
    solver name, so the same physical system can be checked for every unraveling.
    The SME solvers use perfect efficiency (eta=1) and no dark counts (theta=0),
    for which their trajectory-averaged dynamics matches the corresponding SSE.
    """

    @abstractmethod
    def Ls(self, params: PyTree) -> list[QArray | TimeQArray]:
        """Compute the jump operators."""

    @property
    @abstractmethod
    def etas(self) -> Array:
        """Compute the efficiencies for each jump operator."""

    def run(self, solver: str, method: Method, keys: Array) -> Result:
        H, Ls, y0 = self.H(None), self.Ls(None), self.y0(None)
        exp_ops = self.Es(None) or None
        if solver == 'jsse':
            return dq.jssesolve(
                H, Ls, y0, self.tsave, keys=keys, exp_ops=exp_ops, method=method
            )
        if solver == 'dsse':
            return dq.dssesolve(
                H, Ls, y0, self.tsave, keys=keys, exp_ops=exp_ops, method=method
            )
        if solver == 'jsme':
            thetas = jnp.zeros(len(Ls))
            return dq.jsmesolve(
                H,
                Ls,
                thetas,
                self.etas,
                y0,
                self.tsave,
                keys=keys,
                exp_ops=exp_ops,
                method=method,
            )
        if solver == 'dsme':
            return dq.dsmesolve(
                H,
                Ls,
                self.etas,
                y0,
                self.tsave,
                keys=keys,
                exp_ops=exp_ops,
                method=method,
            )
        raise ValueError(f'unknown stochastic solver {solver!r}')


class ProtectedSubspace(StochasticSystem):
    """Two qubits with a measurement that is the identity on the odd-parity
    subspace {|01>, |10>}, so it introduces no back-action. The exact trajectory
    is the deterministic ket cos(wt)|01> - i sin(wt)|10>.
    """

    def __init__(self, *, omega: float, tsave: Array):
        self.n = 4
        self.omega = omega
        self.tsave = tsave
        self.params_default = None

    def H(self, params: PyTree) -> QArray | TimeQArray:  # noqa: ARG002
        sx, sy = dq.sigmax(), dq.sigmay()
        return 0.5 * self.omega * ((sx & sx) + (sy & sy))

    def Ls(self, params: PyTree) -> list[QArray | TimeQArray]:  # noqa: ARG002
        sz = dq.sigmaz()
        return [-(sz & sz)]

    @property
    def etas(self) -> Array:
        return jnp.array([1.0])

    def y0(self, params: PyTree) -> QArray:  # noqa: ARG002
        return dq.fock(2, 0) & dq.fock(2, 1)  # |01>

    def Es(self, params: PyTree) -> list[QArray]:  # noqa: ARG002
        return []

    def state(self, t: float) -> QArray:
        ket01 = dq.fock(2, 0) & dq.fock(2, 1)
        ket10 = dq.fock(2, 1) & dq.fock(2, 0)
        return jnp.cos(self.omega * t) * ket01 - 1j * jnp.sin(self.omega * t) * ket10


class BackactionQubit(ProtectedSubspace):
    """Same Hamiltonian and initial state as `ProtectedSubspace`, but with a loss
    operator L = sz⊗I that is not the identity on the subspace, so the measurement
    does introduce genuine back-action (negative control). `state` is inherited and
    used as the reference the trajectories should deviate from.
    """

    def Ls(self, params: PyTree) -> list[QArray | TimeQArray]:  # noqa: ARG002
        return [dq.sigmaz() & dq.eye(2)]

    @property
    def etas(self) -> Array:
        return jnp.array([1.0])


class DampedOscillator(StochasticSystem):
    """Driven-damped harmonic oscillator. The trajectory-averaged amplitude has the
    known analytical solution <a>(t) = alpha_ss (1 - e^{-s t}), s = kappa/2 + i omega,
    used as the convergence reference (instead of mesolve).
    """

    def __init__(self, *, n: int, omega: float, kappa: float, eps: float, tsave: Array):
        self.n = n
        self.omega = omega
        self.kappa = kappa
        self.eps = eps
        self.tsave = tsave
        self.params_default = None

    def H(self, params: PyTree) -> QArray | TimeQArray:  # noqa: ARG002
        a = dq.destroy(self.n)
        return self.omega * a.dag() @ a + self.eps * (a + a.dag())

    def Ls(self, params: PyTree) -> list[QArray | TimeQArray]:  # noqa: ARG002
        return [
            jnp.sqrt(0.5 * self.kappa) * dq.destroy(self.n),
            jnp.sqrt(0.5 * self.kappa) * dq.destroy(self.n),
        ]

    @property
    def etas(self) -> Array:
        return jnp.array([1.0, 0.0])

    def y0(self, params: PyTree) -> QArray:  # noqa: ARG002
        return dq.coherent(self.n, 0)

    def Es(self, params: PyTree) -> list[QArray]:  # noqa: ARG002
        return [dq.destroy(self.n)]

    def _alpha(self, t: float) -> Array:
        s = self.kappa / 2 + 1j * self.omega
        return -1j * self.eps / s * (1.0 - jnp.exp(-s * t))

    def state(self, t: float) -> QArray:
        # the damped driven oscillator from vacuum stays a coherent state |alpha(t)>
        return dq.coherent(self.n, self._alpha(t))

    def expect(self, t: float) -> Array:
        return jnp.array([self._alpha(t)])


class DecayQubit(StochasticSystem):
    """Spontaneously decaying qubit (H = 0, L = sqrt(gamma) sigma_-, psi0 = |e>). For
    the jump unravelings each trajectory is either still excited or has decayed, so
    the excited population is Bernoulli(e^{-gamma t}).
    """

    def __init__(self, *, gamma: float, tsave: Array):
        self.n = 2
        self.gamma = gamma
        self.tsave = tsave
        self.params_default = None

    def H(self, params: PyTree) -> QArray | TimeQArray:  # noqa: ARG002
        return dq.zeros(2)

    def Ls(self, params: PyTree) -> list[QArray | TimeQArray]:  # noqa: ARG002
        return [jnp.sqrt(self.gamma) * dq.sigmam()]

    @property
    def etas(self) -> Array:
        return jnp.array([1.0])

    def y0(self, params: PyTree) -> QArray:  # noqa: ARG002
        return dq.excited()

    def Es(self, params: PyTree) -> list[QArray]:  # noqa: ARG002
        return [dq.excited().todm()]

    def excited_population(self, t: Array) -> Array:
        return jnp.exp(-self.gamma * t)


class QNDQubit(StochasticSystem):
    """QND measurement of sigma_z on a sigma_z eigenstate (H = 0, L = sqrt(gamma) sz,
    psi0 = |e>). The state is a fixed point, so the only randomness is in the
    measurement record, with analytically known moments: a Poisson(gamma T) click
    count for the jump unravelings, and a Gaussian record (mean 2 sqrt(gamma),
    variance 1/dt) for the diffusive ones.
    """

    def __init__(self, *, gamma: float, tsave: Array):
        self.n = 2
        self.gamma = gamma
        self.tsave = tsave
        self.params_default = None

    def H(self, params: PyTree) -> QArray | TimeQArray:  # noqa: ARG002
        return dq.zeros(2)

    def Ls(self, params: PyTree) -> list[QArray | TimeQArray]:  # noqa: ARG002
        return [jnp.sqrt(self.gamma) * dq.sigmaz()]

    @property
    def etas(self) -> Array:
        return jnp.array([1.0])

    def y0(self, params: PyTree) -> QArray:  # noqa: ARG002
        return dq.excited()

    def Es(self, params: PyTree) -> list[QArray]:  # noqa: ARG002
        return []

    def poisson_lambda(self) -> float:
        return float(self.gamma * (self.tsave[-1] - self.tsave[0]))

    def record_mean(self) -> float:
        sz = dq.expect(dq.sigmaz(), self.y0(None)).real
        return float(2 * jnp.sqrt(self.gamma) * sz)

    def record_variance(self) -> float:
        return float(1.0 / (self.tsave[1] - self.tsave[0]))


# physical systems used by the stochastic solver tests
_tsave = np.linspace(0.0, 1.0, 11)
protected_subspace = ProtectedSubspace(omega=1.0, tsave=_tsave)
backaction_qubit = BackactionQubit(omega=1.0, tsave=_tsave)
damped_oscillator = DampedOscillator(
    n=8, omega=2 * np.pi, kappa=2.0, eps=2.0, tsave=_tsave
)
decay_qubit = DecayQubit(gamma=1.0, tsave=_tsave)
qnd_qubit = QNDQubit(gamma=1.0, tsave=_tsave)
