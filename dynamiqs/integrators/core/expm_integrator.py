from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from dynamiqs._utils import _concatenate_sort
from dynamiqs.result import Saved

from ...utils.quantum_utils.general import expm
from ...utils.vectorization import slindbladian
from .._utils import ispwc
from .abstract_integrator import BaseIntegrator


class ExpmIntegrator(BaseIntegrator):
    class Infos(eqx.Module):
        nsteps: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                # note: expm solvers can make different number of steps between
                # batch elements when batching over PWC objects
                return (
                    f'avg. {self.nsteps.mean():.1f} steps | infos shape'
                    f' {self.nsteps.shape}'
                )
            return f'{self.nsteps} steps'

    def __init__(self, *args):
        super().__init__(*args)

        # check that Hamiltonian is time-independent
        if not ispwc(self.H):
            raise TypeError('Solver `Expm` requires a piece-wise constant Hamiltonian.')

    def _diff_eq_rhs(self, t: float) -> Array:
        raise NotImplementedError

    def collect_saved(self, saved: Saved, ylast: Array, times: Array) -> Saved:
        # === extract the states and expects at the save times ts
        t_idxs = jnp.searchsorted(times[1:], self.ts)
        if self.options.save_states:
            saved = eqx.tree_at(lambda x: x.ysave, saved, saved.ysave[t_idxs])
        if saved.Esave is not None:
            saved = eqx.tree_at(lambda x: x.Esave, saved, saved.Esave[t_idxs])
        if saved.extra is not None:
            saved = eqx.tree_at(lambda x: x.extra, saved, saved.extra[t_idxs])
        return super().collect_saved(saved, ylast)

    def run(self) -> PyTree:
        # === find all times at which to stop in [t0, t1]
        # find all times where the solution should be saved (self.ts) or at which the
        # Hamiltonian changes (self.H.discontinuity_ts)
        disc_ts = self.H.discontinuity_ts
        if disc_ts is not None:
            disc_ts = disc_ts.clip(self.t0, self.t1)
        times = _concatenate_sort(jnp.asarray([self.t0]), self.ts, disc_ts)  # (ntimes,)

        # === compute time differences (null for times outside [t0, t1])
        delta_ts = jnp.diff(times)  # (ntimes-1,)

        # === batch-compute the propagators on each time interval
        Hs = jax.vmap(self._diff_eq_rhs)(times[:-1])  # (ntimes-1, n, n)
        step_propagators = expm(delta_ts[:, None, None] * Hs)  # (ntimes-1, n, n)

        # === combine the propagators together
        def step(carry: Array, x: Array) -> tuple[Array, Array]:
            # note the ordering x @ carry: we accumulate propagators from the left
            U_next = x @ carry
            return U_next, self.save(U_next)

        ylast, saved = jax.lax.scan(step, self.y0, step_propagators)  # (ntimes-1, n, n)

        # === save the propagators
        nsteps = (delta_ts != 0).sum()
        saved = self.collect_saved(saved, ylast, times)
        return self.result(saved, infos=self.Infos(nsteps))


class SEExpmIntegrator(ExpmIntegrator):
    def _diff_eq_rhs(self, t: float) -> Array:
        return -1j * self.H(t)


class MEExpmIntegrator(ExpmIntegrator):
    def _diff_eq_rhs(self, t: float) -> Array:
        return slindbladian(self.H(t), [L(t) for L in self.Ls])
