import diffrax as dx
import jax.numpy as jnp

from ..utils.utils import dag
from .lindblad_term import Hnh, LindbladTerm


class Rouchon1Solver(dx.AbstractSolver):
    term_structure = LindbladTerm
    interpolation_cls = dx._local_interpolation.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        pass

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump

        # compute y1
        dt = terms.contr(t0, t1)
        H_t = terms.term.H(t0)
        Ls_t = jnp.stack([L(t0) for L in terms.term.Ls])
        M0 = terms.term.Id - 1j * dt * Hnh(H_t, Ls_t)  # (n, n)
        M1s = jnp.sqrt(jnp.abs(dt)) * Ls_t  # (nL, n, n)
        y1 = M0 @ y0 @ dag(M0) + jnp.sum(M1s @ y0 @ dag(M1s), axis=0)

        # return
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, dx._solution.RESULTS.successful

    def func(self, terms, t0, y0, args):
        pass
