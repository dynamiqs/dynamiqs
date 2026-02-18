# ruff: noqa: ANN001, ANN201, ARG002
# we mostly ignore type hinting in this file for readability purposes

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import reduce
from itertools import product

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax._custom_types import RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax import diffeqsolve, SaveAt, Euler, Midpoint, Bosh3, Tsit5, ConstantStepSize, ODETerm


from ...qarrays.qarray import QArray
from ...time_qarray import ConstantTimeQArray
from ...utils.operators import asqarray, eye_like
from .diffrax_integrator import MESolveDiffraxIntegrator

from ...qarrays.layout import dense, dia



class AbstractRouchonTerm(dx.AbstractTerm):
    # this class bypasses the typical Diffrax term implementation, as Rouchon schemes
    # don't match the vf/contr/prod structure

    rouchon_step: Callable[[RealScalarLike, RealScalarLike, Y], [Y, Y]]
    # should be defined as `rouchon_step(t0, t1, y0) -> y1, error`

    def vf(self, t, y, args):
        pass

    def contr(self, t0, t1, **kwargs):
        pass

    def prod(self, vf, control):
        pass


class RouchonDXSolver(dx.AbstractSolver):
    _order: int
    term_structure = AbstractRouchonTerm
    interpolation_cls = LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        pass

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        y1, error = terms.term.rouchon_step(t0, t1, y0)
        dense_info = dict(y0=y0, y1=y1)
        return y1, error, dense_info, None, dx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        pass

    def order(self, terms):
        return self._order


class AdaptiveRouchonDXSolver(dx.AbstractAdaptiveSolver, RouchonDXSolver):
    pass


class KrausChannel(eqx.Module):
    operators: list[QArray]

    def __call__(self, rho) -> QArray:
        return sum([M @ rho @ M.dag() for M in self.operators])

    def apply_conjugate(self, op) -> QArray:
        # computes S = sum(M† op M) for the channel
        return sum([M.dag() @ op @ M for M in self.operators])

    def S(self) -> QArray:
        # computes S = sum(M†M) for the channel
        return sum([M.dag() @ M for M in self.operators])

    def get_kraus_operators(self) -> list[QArray]:
        # returns the list of Kraus operators in the channel.
        return self.operators


class NestedKrausChannel(eqx.Module):
    channels: list[KrausChannel]

    def __init__(self, *channels: KrausChannel):
        self.channels = list(channels)

    def __call__(self, rho) -> QArray:
        for channel in reversed(self.channels):
            rho = channel(rho)
        return rho

    def S(self) -> QArray:
        # computes S = sum(M†M) for the nested channel
        if len(self.channels) == 0:
            return eye_like(self.channels[0].operators[0])

        out = self.channels[0].S()
        for channel in self.channels[1:]:
            out = channel.apply_conjugate(out)
        return out

    def get_kraus_operators(self) -> list[QArray]:
        # returns the list of all Kraus operators in the nested channel.
        return [
            reduce(lambda a, b: a @ b, x)  # product of all the operators
            for x in product(*[ch.operators for ch in self.channels])
        ]


class KrausMap(eqx.Module):
    channels: list[NestedKrausChannel | KrausChannel]

    def __init__(self, *channels: NestedKrausChannel | KrausChannel):
        self.channels = list(channels)

    def __call__(self, rho) -> QArray:
        return sum([channel(rho) for channel in self.channels])

    def S(self) -> QArray:
        # computes S = sum(M†M) for the full map
        return sum([channel.S() for channel in self.channels])

    def get_kraus_operators(self) -> list[QArray]:
        # returns the list of all Kraus operators in the map.
        return [op for c in self.channels for op in c.get_kraus_operators()]

class RKStage(eqx.Module):
    no_jump_0: QArray
    no_jump_i: QArray
    Ls: list[Sequence[QArray]]
    dt: float
    aij: float

    def __call__(self, rho0, rhoi) -> list[QArray]:
        return self.no_jump_0@rho0@self.no_jump_0.dag() + self.no_jump_i @ (self.dt* self.aij * sum([_L @ rhoi @ _L.dag() for _L in self.Ls]))@ self.no_jump_i.dag()
    
    def S(self, O):
        return self.S_nojump(O) + self.S_jump(O)
    
    def S_nojump(self, O):
        # Contribution from no_jump_0 operator: M0† @ O @ M0
        return self.no_jump_0.dag() @ O @ self.no_jump_0
    
    def S_jump(self, O):
        # Contribution from jump operators: sum(Mi† @ O @ Mi) for i >= 1
        O_sandwiched = self.no_jump_i.dag() @ O @ self.no_jump_i
        return self.dt * self.aij * sum([_L.dag() @ O_sandwiched @ _L for _L in self.Ls])
    
    def S_composed(self, O, prev_stage_S):
        # For add_kraus_operators: no_jump_0 is standalone, jump ops are composed
        # sum(M† @ O @ M) = no_jump_0† @ O @ no_jump_0 + prev_stage_S(S_jump(O))
        return self.S_nojump(O) + prev_stage_S(self.S_jump(O))
    
    def get_kraus_operators(self):
        return [self.no_jump_0] + [jnp.sqrt(self.dt * self.aij) * self.no_jump_i @ _L for _L in self.Ls]
    
    def add_kraus_operators(self, kraus_ops) -> list[QArray]:
        res_int = [jnp.sqrt(self.dt * self.aij) * self.no_jump_i @ _L for _L in self.Ls]
        return [self.no_jump_0] + [op_int @ op_stage for op_int, op_stage in product(res_int, kraus_ops)]

class FirstStage(RKStage): # Identity stage
    def __call__(self, rho0, _rhom) -> list[QArray]:
        return rho0
    
    def S(self, O):
        return O
    
    def S_nojump(self, O):
        return O
    
    def S_jump(self, O):
        return 0 * O  # Zero contribution
    
    def S_composed(self, O, prev_stage_S):
        return O

class SecondStage(RKStage): # if no_jump_i is equal to no_jump_0
    def __call__(self, rho0, rhoi) -> list[QArray]:
        return self.no_jump_0 @ (rho0 + self.dt* self.aij * sum([_L @ rhoi @ _L.dag() for _L in self.Ls]))@ self.no_jump_0.dag()
    
    def S_nojump(self, O):
        return self.no_jump_0.dag() @ O @ self.no_jump_0
    
    def S_jump(self, O):
        O_sandwiched = self.no_jump_0.dag() @ O @ self.no_jump_0
        return self.dt * self.aij * sum([_L.dag() @ O_sandwiched @ _L for _L in self.Ls])
    
    def S(self, O):
        return self.S_nojump(O) + self.S_jump(O)
    
class SameTimeStage(RKStage): #if no_jump_i is identity
    def __call__(self, rho0, rhom) -> list[QArray]:
        return self.no_jump_0 @ rho0 @ self.no_jump_0.dag() + self.dt* self.aij * sum([_L @ rhom @ _L.dag() for _L in self.Ls])
    
    def S_nojump(self, O):
        return self.no_jump_0.dag() @ O @ self.no_jump_0
    
    def S_jump(self, O):
        # no_jump_i is identity, so no sandwich
        return self.dt * self.aij * sum([_L.dag() @ O @ _L for _L in self.Ls])
    
    def S(self, O):
        return self.S_nojump(O) + self.S_jump(O)
    
class KrausRK(eqx.Module):
    no_jump_propagator: Callable[[RealScalarLike], QArray]
    t: RealScalarLike
    dt: RealScalarLike
    Ls: Callable[[RealScalarLike], Sequence[QArray]]
    identity: QArray   

class KrausEuler(KrausRK):
    @property
    def nojump_0to1(self):
        return self.no_jump_propagator(self.t + self.dt)

    def __call__(self, rho0) -> list[QArray]:
        return self.nojump_0to1 @ rho0 @ self.nojump_0to1.dag() + self.dt * sum([_L @ rho0 @ _L.dag() for _L in self.Ls(self.t)])
    def S(self):
        return self.nojump_0to1.dag() @ self.nojump_0to1 + self.dt * sum([_L.dag() @ _L for _L in self.Ls(self.t)])
    
    def get_kraus_operators(self):
        return [self.nojump_0to1] + [jnp.sqrt(self.dt) * _L for _L in self.Ls(self.t)]

class KrausHeun2(KrausRK): #Heun's second order method    
    @property
    def nojump_0to1(self):
        return self.no_jump_propagator(self.t + self.dt)

    @property
    def Ls0(self):
        return self.Ls(self.t)
    
    @property
    def Ls1(self):
        return self.Ls(self.t + self.dt)

    @property
    def stage2(self):
        return SecondStage(
            no_jump_0=self.nojump_0to1,
            no_jump_i=self.nojump_0to1,
            Ls=self.Ls0,
            dt=self.dt,
            aij = 1,
        )
    
    def __call__(self, rho0) -> list[QArray]:
        rho1 = rho0
        rho2 = self.stage2(rho0, rho1)
        return (self.nojump_0to1 @ (rho0 + self.dt/2*sum(_L@rho1@_L.dag() for _L in self.Ls0))@self.nojump_0to1.dag() 
                + (self.dt/2*sum(_L@rho2@_L.dag() for _L in self.Ls1)))
    
    def S(self):
        O1 = self.nojump_0to1.dag() @ self.nojump_0to1
        O10 = sum(_L.dag() @ O1 @ _L for _L in self.Ls0)
        O11 = sum(_L.dag() @ _L for _L in self.Ls1)
        return (O1 
                + self.dt/2*(O10
                            + self.stage2.S(O11)))
    
    def get_kraus_operators(self):
        return ([self.nojump_0to1]
                + [jnp.sqrt(self.dt/2) * self.nojump_0to1 @ _L for _L in self.Ls0] 
                + [jnp.sqrt(self.dt/2) * _L @ op for _L, op in product(self.Ls1, self.stage2.get_kraus_operators())])

class KrausHeun3(KrausRK): #Heun's third order method
    @property
    def nojump_0to1(self):
        return self.no_jump_propagator(self.t + self.dt)
    
    @property
    def nojump_0to1o3(self):
        return self.no_jump_propagator(self.t + 1/3 * self.dt)

    @property
    def nojump_0to2o3(self):
        return self.no_jump_propagator(self.t + 2/3 * self.dt)
    
    @property
    def nojump_2o3to1(self):
        return solve_propagator(self.no_jump_propagator(self.t + self.dt),
                                 self.no_jump_propagator(self.t + 2/3 * self.dt))
    
    @property
    def nojump_1o3to2o3(self):
        return solve_propagator(self.no_jump_propagator(self.t + 2/3 * self.dt),
                                 self.no_jump_propagator(self.t + 1/3 * self.dt))
    @property
    def Ls0(self):
        return self.Ls(self.t)
    
    @property
    def Ls1o3(self):
        return self.Ls(self.t + 1/3 * self.dt)
    
    @property
    def Ls2o3(self):
        return self.Ls(self.t + 2/3 * self.dt)
    
    @property
    def stage2(self):
        return SecondStage(
            no_jump_0=self.nojump_0to1o3,
            no_jump_i=self.nojump_0to1o3,
            Ls=self.Ls0,
            dt=self.dt,
            aij = 1/3,
        )
    
    @property
    def stage3(self):
        return RKStage(
            no_jump_0=self.nojump_0to2o3,
            no_jump_i=self.nojump_1o3to2o3,
            Ls=self.Ls1o3,
            dt=self.dt,
            aij = 2/3,
        )
    
    def __call__(self, rho0) -> list[QArray]:
        rho1 = rho0
        rho2 = self.stage2(rho0, rho1)
        rho3 = self.stage3(rho0, rho2)
        return (self.nojump_0to1 @ (rho0 + self.dt/4*sum(_L@rho1@_L.dag() for _L in self.Ls0))@self.nojump_0to1.dag() 
                + self.nojump_2o3to1 @ (3*self.dt/4*sum(_L@rho3@_L.dag() for _L in self.Ls2o3))@self.nojump_2o3to1.dag())
    
    def S(self):
        O1 = self.nojump_0to1.dag() @ self.nojump_0to1
        O2 = sum(_L.dag() @ O1 @ _L for _L in self.Ls0)
        O3_nojump = self.nojump_2o3to1.dag() @ self.nojump_2o3to1
        O3 = sum(_L.dag() @ O3_nojump @ _L for _L in self.Ls2o3)
        # For composed operators: stage3's no-jump is standalone, stage3's jump is composed with stage2
        return (O1 
                + self.dt*(1/4*O2
                + 3/4*self.stage3.S_composed(O3, self.stage2.S)))
    
    def get_kraus_operators(self):
        return ([self.nojump_0to1] 
                + [jnp.sqrt(self.dt/4) * self.nojump_0to1 @ _L for _L in self.Ls0]
                + [jnp.sqrt(3*self.dt/4) * self.nojump_2o3to1 @ _L @ op 
                   for _L, op in product(self.Ls2o3,
                                         self.stage3.add_kraus_operators(self.stage2.get_kraus_operators()))])

class KrausRK4(KrausRK): #Optimized RK4 scheme
    @property
    def nojump_0to1(self):
        return self.no_jump_propagator(self.t + self.dt)
    
    @property
    def nojump_0tomid(self):
        return self.no_jump_propagator(self.t + 0.5 * self.dt)
    
    @property
    def nojump_midto1(self):
        return solve_propagator(self.no_jump_propagator(self.t + self.dt), 
                                self.no_jump_propagator(self.t + 0.5 * self.dt))
    
    @property
    def Ls0(self):
        return self.Ls(self.t)
    
    @property
    def Ls1(self):
        return self.Ls(self.t + self.dt)
    
    @property
    def Lsmid(self):
        return self.Ls(self.t + 0.5 * self.dt)

    @property
    def stage_2(self):
        return SecondStage(
            no_jump_0=self.nojump_0tomid,
            no_jump_i=self.nojump_0tomid,
            Ls=self.Ls0,
            dt=self.dt,
            aij = 0.5,
        )
    
    @property
    def stage_3(self):
        return SameTimeStage(
            no_jump_0=self.nojump_0tomid,
            no_jump_i=self.identity,
            Ls=self.Lsmid,
            dt=self.dt,
            aij = 0.5,
        )
    
    @property
    def stage_4(self):
        return RKStage(
            no_jump_0=self.nojump_0to1,
            no_jump_i= self.nojump_midto1,
            Ls=self.Lsmid,
            dt=self.dt,
            aij = 1.0,
        )
    def __call__(self, rho0) -> list[QArray]:
        rho1 = rho0
        rho2 = self.stage_2(rho0, rho1)
        rho3 = self.stage_3(rho0, rho2)
        rho23 = rho2 + rho3
        rho4 = self.stage_4(rho0, rho3)
        return (self.nojump_0to1 @ (rho0 + self.dt/6*sum(_L@rho1@_L.dag() for _L in self.Ls0))@self.nojump_0to1.dag() 
                + self.nojump_midto1 @ (self.dt/3*sum(_L@rho23@_L.dag() for _L in self.Lsmid)) @ self.nojump_midto1.dag()
                + (self.dt/6*sum(_L@rho4@_L.dag() for _L in self.Ls1)))
    
    def S(self): # Applies the map in reverse to the identity
        O1 = self.nojump_0to1.dag() @ self.nojump_0to1
        O2 = sum(_L.dag() @ O1 @ _L for _L in self.Ls0)
        O3_nojump = self.nojump_midto1.dag() @ self.nojump_midto1
        O3 = sum(_L.dag() @ O3_nojump @ _L for _L in self.Lsmid)
        O4 = sum(_L.dag() @ _L for _L in self.Ls1)
        # k2s: dt/3 * stage_2.S(O3)
        # k3s: dt/3 * stage_3.S_composed(O3, stage_2.S)
        # k4s: dt/6 * stage_4.S_composed(O4, lambda X: stage_3.S_composed(X, stage_2.S))
        return (O1 
                + self.dt/6*(O2
                            + 2*self.stage_2.S(O3)
                            + 2*self.stage_3.S_composed(O3, self.stage_2.S)
                            + self.stage_4.S_composed(O4, lambda X: self.stage_3.S_composed(X, self.stage_2.S))))
    
    def get_kraus_operators(self):
        k0s = [self.nojump_0to1]
        k1s = [jnp.sqrt(self.dt/6) * self.nojump_0to1 @ _L for _L in self.Ls0]
        k23s_int = [jnp.sqrt(self.dt/3) * self.nojump_midto1 @ _L for _L in self.Lsmid] # to do less scalar matrix multiplications
        k2s = [k23_int @ op for k23_int, op in product(k23s_int, self.stage_2.get_kraus_operators())]
        k3s = [k23_int @ op for k23_int, op in product(k23s_int, self.stage_3.add_kraus_operators(self.stage_2.get_kraus_operators()))]
        k4s = [jnp.sqrt(self.dt/6) * _L @ op 
               for _L, op in product(self.Ls1, 
                                    self.stage_4.add_kraus_operators(self.stage_3.add_kraus_operators(self.stage_2.get_kraus_operators())))]
        return k0s + k1s + k2s + k3s + k4s
    


def cholesky_normalize(kraus_map: KrausMap, rho: QArray) -> jax.Array:
    # To normalize the scheme, we compute
    #   S = sum_k Mk^† @ Mk
    # and replace
    #   Mk by ~Mk = Mk @ S^{-1/2}
    # such that
    #   sum_k ~Mk^† @ ~Mk = S^{†(-1/2)} @ (sum_k Mk^† @ Mk) @ S^{-1/2}
    #                   = S^{†(-1/2)} @ S @ S^{-1/2}
    #                   = I
    # To (i) keep sparse matrices and (ii) have a generic implementation that also
    # works for time-dependent systems, we use a Cholesky decomposition at each step
    # instead of computing S^{-1/2} explicitly. We write S = T @ T^† with T lower
    # triangular, and we replace
    #   Mk by ~Mk = Mk @ S^{-1/2}
    # such that
    #   #   sum_k ~Mk^† @ ~Mk = T^{-1} @ (sum_k Mk^† @ Mk) @ T^{†(-1)}
    #                       = T^{-1} @ T @ T^† @ T^{(-1)}
    #                       = I
    # In practice we directly replace rho_k by T^{†(-1)} @ rho_k @ T^{-1} instead of
    # computing all ~Mks.

    S = kraus_map.S()
    T = jnp.linalg.cholesky(S.to_jax())  # T lower triangular

    # we want T^{†(-1)} @ y0 @ T^{-1}
    rho = rho.to_jax()
    # solve T^† @ x = rho => x = T^{†(-1)} @ rho
    rho = jax.lax.linalg.triangular_solve(
        T, rho, lower=True, transpose_a=True, conjugate_a=True
    )
    # solve x @ T = rho => x = rho @ T^{-1}
    return jax.lax.linalg.triangular_solve(T, rho, lower=True, left_side=True)


def order2_nojump_evolution(
    H: Callable[[RealScalarLike], QArray],
    L: Callable[[RealScalarLike], Sequence[QArray]],
    t: float,
    dt: float,
) -> Callable[[float], QArray]:
    """Evaluates the no-jump evolution between t and t+dt
    using the explicit midpoint method.
    """
    G0 = -1j * H(t) - 0.5 * sum([_L.dag() @ _L for _L in L(t)])
    Gmid = -1j * H(t + 0.5 * dt) - 0.5 * sum([_L.dag() @ _L for _L in L(t + 0.5 * dt)])
    U0 = eye_like(G0)
    k1 = U0 + G0 * dt / 2
    return U0 + dt * Gmid @ k1


def order3_nojump_dense_evolution(
    H: Callable[[RealScalarLike], QArray],
    L: Callable[[RealScalarLike], Sequence[QArray]],
    t: float,
    dt: float,
) -> Callable[[float], QArray]:
    """Evaluates the no-jump evolution between t and t+dt
    using Kutta's third order method with dense output.
    """
    G0 = -1j * H(t) - 0.5 * sum([_L.dag() @ _L for _L in L(t)])
    Gmid = -1j * H(t + dt / 2) - 0.5 * sum([_L.dag() @ _L for _L in L(t + dt / 2)])
    G1 = -1j * H(t + dt) - 0.5 * sum([_L.dag() @ _L for _L in L(t + dt)])
    U0 = eye_like(G0)
    k1 = G0
    k2 = Gmid @ (U0 + (dt / 2) * k1)
    k3 = G1 @ (U0 - dt * k1 + 2 * dt * k2)
    U1 = U0 + dt / 6 * (k1 + 4 * k2 + k3)

    def interp(s: float) -> QArray:
        # Quadratic Hermite interpolation: p(theta) = a0 + a1*theta + a2*theta^2
        # Constraints: p(0)=U0, p(1)=U1, p'(0)=dt*f0
        theta = (s - t) / dt
        a0 = U0
        a1 = dt * k1
        a2 = U1 - U0 - dt * k1
        return a0 + theta * a1 + theta**2 * a2

    return interp

def solve_propagator(U1, U2) -> QArray:
    """Compute the no jump propagator from t2 to t1 using LU factorization.

    U1: propagator from 0 to t1
    U2: propagator from 0 to t2
    Returns: propagator from t2 to t1
    """
    # U1 = U(t2->t1) @ U2, so U(t2->t1) = U1 @ U2^{-1}
    # Compute U2^{-1} using LU factorization
    return asqarray(jnp.linalg.solve(U2.to_jax().T, U1.to_jax().T).T, dims=U1.dims)


class MESolveFixedRouchonIntegrator(MESolveDiffraxIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using a
    fixed step Rouchon method.
    """

    @property
    def time_dependent(self) -> bool:
        return not isinstance(self.H, ConstantTimeQArray)
    
    @property
    def G(self):
        def G_at_t(t):
            LdL = sum([_L.dag() @ _L for _L in self.L(t)])
            return -1j * self.H(t) - 0.5 * LdL
        return G_at_t

    @property
    def identity(self):
        return eye_like(self.H(0), layout = dense)
    
    @property
    def no_jump_solver(self):
        return Euler()
    
    @property
    def terms(self) -> dx.AbstractTerm:
        def rouchon_step(t0, t1, y0):  # noqa: ANN202
            # The Rouchon update for a single loss channel is:
            #   rho_{k+1} = sum_k Mk @ rho_k @ Mk^†
            # See comment of `cholesky_normalize()` for the normalization.

            rho = y0
            dt = t1 - t0
            kraus_map = self._build_kraus_map(t0, dt)

            if self.method.normalize:
                rho = cholesky_normalize(kraus_map, rho)

            # for fixed step size, we return None for the error estimate
            return kraus_map(rho), None

        return AbstractRouchonTerm(rouchon_step)
    
    @property
    def no_jump_propagator(self):
        def _no_jump_propagator_flow(t, y, *args):
            return self.G(t) @ y
        no_jump_propagator_flow = ODETerm(_no_jump_propagator_flow)
        def _no_jump_propagator(t, dt):
            solver = self.no_jump_solver
            solver_state = solver.init(no_jump_propagator_flow, t, t + dt, self.identity, None)
            y1, error, dense_info, solver_state, result = solver.step(
                no_jump_propagator_flow,
                t0=t,
                t1=t + dt,
                y0=self.identity,
                args=None,
                solver_state=solver_state,
                made_jump=False,
            )
            interpolant = solver.interpolation_cls(
                t0=t,
                t1=t + dt,
                **dense_info,
            )
            return interpolant.evaluate
        return _no_jump_propagator

    def _build_kraus_map(self, t: float, dt: float) -> RK:
        return self.build_kraus_map(self.no_jump_propagator(t, dt), self.L, t, dt, self.identity)

    @staticmethod
    @abstractmethod
    def build_kraus_map(
        no_jump_propagator: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]], 
        t: RealScalarLike, 
        dt: RealScalarLike, 
        identity: QArray
    ) -> RK:
        pass

    


class MESolveFixedRouchon1Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 1 method.
    """
    @property
    def no_jump_solver(self):
        return Euler()

    @staticmethod
    def build_kraus_map(
        no_jump_propagator: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: RealScalarLike,
        dt: RealScalarLike,
        identity: QArray) -> KrausRK:
        return KrausEuler(
            no_jump_propagator=no_jump_propagator,
            t=t,
            dt=dt,
            Ls=L,
            identity=identity)


mesolve_rouchon1_integrator_constructor = lambda **kwargs: (
    MESolveFixedRouchon1Integrator(
        **kwargs, diffrax_solver=RouchonDXSolver(1), fixed_step=True
    )
)


class MESolveFixedRouchon2Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 2 method.
    """
    @property
    def no_jump_solver(self):
        return Midpoint()

    @staticmethod
    def build_kraus_map(
        no_jump_propagator: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: RealScalarLike,
        dt: RealScalarLike,
        identity: QArray,
    ) -> KrausRK:
        return KrausHeun2(
            no_jump_propagator=no_jump_propagator,
            t=t,
            dt=dt,
            Ls=L,
            identity=identity,
        )

class MESolveFixedRouchon3Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 3 method.
    """

    @property
    def no_jump_solver(self):
        return Bosh3()

    @staticmethod
    def build_kraus_map(
        no_jump_propagator: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: RealScalarLike,
        dt: RealScalarLike,
        identity: QArray,
    ) -> KrausRK:
        return KrausHeun3(
            no_jump_propagator=no_jump_propagator,
            t=t,
            dt=dt,
            Ls=L,
            identity=identity,
        )


class MESolveAdaptiveRouchonIntegrator(MESolveDiffraxIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using an
    adaptive Rouchon method.
    """

    @property
    def G(self):
        def G_at_t(t):
            LdL = sum([_L.dag() @ _L for _L in self.L(t)])
            return -1j * self.H(t) - 0.5 * LdL
        return G_at_t

    @property
    def identity(self):
        return eye_like(self.H(0), layout=dense)
    
    @property
    def no_jump_solver_low(self):
        pass
    
    @property
    def no_jump_solver_high(self):
        pass
        
    @property
    def time_dependent(self) -> bool:
        return not isinstance(self.H, ConstantTimeQArray)
    
    @property
    def no_jump_propagators(self):
        no_jump_propagator_term = ODETerm(lambda t, y, args: self.G(t) @ y)

        def _no_jump_propagator_low(t, dt):
            solver_low = self.no_jump_solver_low
            state_low = solver_low.init(no_jump_propagator_term, t, t + dt, self.identity, None)
            y1, error, dense_info, state_low, result = solver_low.step(
                no_jump_propagator_term,
                t0=t,
                t1=t + dt,
                y0=self.identity,
                args=None,
                solver_state=state_low,
                made_jump=False,
            )
            interpolant_low = solver_low.interpolation_cls(
                t0=t,
                t1=t + dt,
                **dense_info,
            )
            return interpolant_low.evaluate

        def _no_jump_propagator_high(t, dt):
            solver_high = self.no_jump_solver_high
            state_high = solver_high.init(no_jump_propagator_term, t, t + dt, self.identity, None)
            y1, error, dense_info, state_high, result = solver_high.step(
                no_jump_propagator_term,
                t0=t,
                t1=t + dt,
                y0=self.identity,
                args=None,
                solver_state=state_high,
                made_jump=False,
            )
            interpolant_high = solver_high.interpolation_cls(
                t0=t,
                t1=t + dt,
                **dense_info,
            )
            return interpolant_high.evaluate
        return _no_jump_propagator_low, _no_jump_propagator_high

    @property
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        # todo: can we do better?
        stepsize_controller = super().stepsize_controller
        # fix incorrect default linear interpolation by stepping exactly at all times
        # in tsave, so interpolation is bypassed
        return replace(stepsize_controller, step_ts=self.ts)


class MESolveAdaptiveRouchon2Integrator(MESolveAdaptiveRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    adaptive Rouchon 1-2 method with embedded dense outputs from Midpoint.
    """

    @property
    def no_jump_propagators(self):
        """Returns embedded order 1 (Euler) and order 2 (Midpoint) propagators
        from a single Midpoint computation, using the embedded error estimate.
        """
        def _no_jump_propagator_flow(t, y, *args):
            return self.G(t) @ y
        no_jump_propagator_term = ODETerm(_no_jump_propagator_flow)
        solver_low = Euler()
        solver_high = Midpoint()

        def _no_jump_propagators(t, dt):
            y0 = self.identity

            # Run Midpoint step to get the order 2 result and the embedded error
            solver_state = solver_high.init(no_jump_propagator_term, t, t + dt, y0, None)
            y1_high, error, dense_info_high, solver_state, result = solver_high.step(
                no_jump_propagator_term,
                t0=t,
                t1=t + dt,
                y0=y0,
                args=None,
                solver_state=solver_state,
                made_jump=False,
            )

            # Midpoint error = y_high - y_low (order 2 minus embedded order 1)
            # So y_low = y_high - error
            y1_low = y1_high - error
            dense_info_low = dict(y0=y0, y1=y1_low)

            # Create interpolants using each solver's interpolation class
            interpolant_low = solver_low.interpolation_cls(
                t0=t,
                t1=t + dt,
                **dense_info_low,
            )
            interpolant_high = solver_high.interpolation_cls(
                t0=t,
                t1=t + dt,
                **dense_info_high,
            )

            return interpolant_low.evaluate, interpolant_high.evaluate

        return _no_jump_propagators

    @property
    def terms(self) -> dx.AbstractTerm:
        def rouchon_step(t0, t1, y0):  # noqa: ANN202
            rho = y0
            dt = t1 - t0
            no_jump_propagator_low, no_jump_propagator_high = self.no_jump_propagators(t0, dt)
            # === first order
            kraus_map_1 = MESolveFixedRouchon1Integrator.build_kraus_map(
                no_jump_propagator_low, self.L, t0, dt, self.identity
            )
            rho_1 = (
                cholesky_normalize(kraus_map_1, rho) if self.method.normalize else rho
            )
            rho_1 = kraus_map_1(rho_1)

            # === second order
            kraus_map_2 = MESolveFixedRouchon2Integrator.build_kraus_map(
                no_jump_propagator_high, self.L, t0, dt, self.identity
            )
            rho_2 = (
                cholesky_normalize(kraus_map_2, rho) if self.method.normalize else rho
            )
            rho_2 = kraus_map_2(rho_2)

            return rho_2, 0.5 * (rho_2 - rho_1)

        return AbstractRouchonTerm(rouchon_step)


class MESolveAdaptiveRouchon3Integrator(MESolveAdaptiveRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    adaptive Rouchon 2-3 method with embedded dense outputs from Bosh3.
    """

    @property
    def no_jump_propagators(self):
        """Returns embedded order 2 and order 3 propagators from a single Bosh3 
        computation, using the embedded error estimate.
        """
        def _no_jump_propagator_flow(t, y, *args):
            return self.G(t) @ y
        no_jump_propagator_term = ODETerm(_no_jump_propagator_flow)
        solver_low = Midpoint()
        solver_high = Bosh3()

        def _no_jump_propagators(t, dt):
            y0 = self.identity

            # Run Bosh3 step to get the order 3 result and the embedded error
            solver_state = solver_high.init(no_jump_propagator_term, t, t + dt, y0, None)
            y1_high, error, dense_info_high, solver_state, result = solver_high.step(
                no_jump_propagator_term,
                t0=t,
                t1=t + dt,
                y0=y0,
                args=None,
                solver_state=solver_state,
                made_jump=False,
            )

            # Bosh3 error = y_high - y_low (order 3 minus embedded order 2)
            # So y_low = y_high - error
            y1_low = y1_high - error

            # dense_info_high contains k (the stages), extract k0 and k1 for Midpoint interpolation
            k = dense_info_high['k']
            dense_info_low = dict(y0=y0, y1=y1_low, k=k[:2])

            # Create interpolants using each solver's interpolation class
            interpolant_low = solver_low.interpolation_cls(
                t0=t,
                t1=t + dt,
                **dense_info_low,
            )
            interpolant_high = solver_high.interpolation_cls(
                t0=t,
                t1=t + dt,
                **dense_info_high,
            )

            return interpolant_low.evaluate, interpolant_high.evaluate

        return _no_jump_propagators

    @property
    def terms(self) -> dx.AbstractTerm:
        def rouchon_step(t0, t1, y0):  # noqa: ANN202
            rho = y0
            dt = t1 - t0
            
            no_jump_propagator_low, no_jump_propagator_high = self.no_jump_propagators(t0, dt)
            
            # === second order
            kraus_map_2 = MESolveFixedRouchon2Integrator.build_kraus_map(
                no_jump_propagator_low, self.L, t0, dt, self.identity
            )
            rho_2 = (
                cholesky_normalize(kraus_map_2, rho) if self.method.normalize else rho
            )
            rho_2 = kraus_map_2(rho_2)

            # === third order
            kraus_map_3 = MESolveFixedRouchon3Integrator.build_kraus_map(
                no_jump_propagator_high, self.L, t0, dt, self.identity
            )
            rho_3 = (
                cholesky_normalize(kraus_map_3, rho) if self.method.normalize else rho
            )
            rho_3 = kraus_map_3(rho_3)
            return rho_3, 0.5 * (rho_3 - rho_2)

        return AbstractRouchonTerm(rouchon_step)


def mesolve_rouchon2_integrator_constructor(**kwargs):
    """Factory function to create a Rouchon2 integrator."""
    if kwargs['method'].dt is not None:
        return MESolveFixedRouchon2Integrator(
            **kwargs, diffrax_solver=RouchonDXSolver(2), fixed_step=True
        )
    return MESolveAdaptiveRouchon2Integrator(
        **kwargs, diffrax_solver=AdaptiveRouchonDXSolver(2), fixed_step=False
    )


def mesolve_rouchon3_integrator_constructor(**kwargs):
    """Factory function to create a Rouchon3 integrator."""
    if kwargs['method'].dt is not None:
        return MESolveFixedRouchon3Integrator(
            **kwargs, diffrax_solver=RouchonDXSolver(3), fixed_step=True
        )
    return MESolveAdaptiveRouchon3Integrator(
        **kwargs, diffrax_solver=AdaptiveRouchonDXSolver(3), fixed_step=False
    )
