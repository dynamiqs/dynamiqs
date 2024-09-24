# ruff: noqa: ANN001, ANN201, ARG002
# we mostly ignore type hinting in this file for readability purposes

from __future__ import annotations

import diffrax as dx
import jax.numpy as jnp
from diffrax._custom_types import RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation

from ...qarrays import QArray
from ..core.abstract_integrator import MESolveIntegrator
from ..core.diffrax_integrator import FixedStepDiffraxIntegrator
from ..core.save_mixin import SolveSaveMixin


class AbstractRouchonTerm(dx.AbstractTerm):
    # this class bypasses the typical Diffrax term implementation, as Rouchon schemes
    # don't match the vf/contr/prod structure

    kraus_map: callable[[RealScalarLike, RealScalarLike, Y], Y]
    # should be defined as `kraus_map(t0, t1, y0) -> y1`

    def vf(self, t, y, args):
        pass

    def contr(self, t0, t1, **kwargs):
        pass

    def prod(self, vf, control):
        pass


class RouchonDXSolver(dx.AbstractSolver):
    term_structure = AbstractRouchonTerm
    interpolation_cls = LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        pass

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        y1 = terms.term.kraus_map(t0, t1, y0)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, dx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        pass


class Rouchon1DXSolver(RouchonDXSolver):
    def order(self, terms):
        return 1


class MESolveRouchon1Integrator(
    FixedStepDiffraxIntegrator, MESolveIntegrator, SolveSaveMixin
):
    """Integrator computing the time evolution of the Lindblad master equation using the
    Rouchon 1 solver.
    """

    diffrax_solver = Rouchon1DXSolver()

    @property
    def Id(self) -> QArray:
        return jnp.eye(self.H.shape[-1], dtype=self.H.dtype)

    @property
    def terms(self) -> dx.AbstractTerm:
        def kraus_map(t0, t1, y0):  # noqa: ANN202
            # The Rouchon update for a single loss channel is:
            #   rho_{k+1} = M0 @ rho @ M0d + M1 @ rho @ M1d
            # with
            #   M0 = I - (iH + 0.5 Ld @ L) dt
            #   M1 = L sqrt(dt)

            delta_t = t1 - t0
            Ls = [L(t0) for L in self.Ls]
            Lsdag = [L.dag() for L in Ls]

            # === compute M0
            LdagL = [Ldag @ L for L, Ldag in zip(Ls, Lsdag)]
            M0 = self.Id - (1j * self.H(t0) + 0.5 * sum(LdagL)) * delta_t

            # === compute Mk
            Mks = [delta_t * L @ y0 @ Ldag for L, Ldag in zip(Ls, Lsdag)]

            return M0 @ y0 @ M0.dag() + sum(Mks)

        return AbstractRouchonTerm(kraus_map)
