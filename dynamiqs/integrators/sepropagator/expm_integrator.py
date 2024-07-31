from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from ..._utils import _concatenate_sort
from ...result import Saved
from ...utils.utils.general import expm
from .._utils import ispwc
from ..core.abstract_integrator import SEPropagatorIntegrator


class SEPropagatorExpmIntegrator(SEPropagatorIntegrator):
    def __init__(self, *args):
        super().__init__(*args)

        # check that Hamiltonian is constant or piecewise constant, or a sum of
        # such Hamiltonians
        if not ispwc(self.H):
            raise TypeError(
                'Solver `Expm` requires a constant or piecewise constant'
                ' Hamiltonian.'
            )

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
        Hs = jax.vmap(self.H)(times[:-1])  # (ntimes-1, n, n)
        step_propagators = expm(-1j * delta_ts[:, None, None] * Hs)  # (ntimes-1, n, n)

        # === combine the propagators together
        def step(carry: Array, x: Array) -> tuple[Array, Array]:
            # note the ordering x @ carry: we accumulate propagators from the left
            U_next = x @ carry
            return U_next, U_next

        _, Us = jax.lax.scan(step, self.y0, step_propagators)  # (ntimes-1, n, n)

        # === extract the propagators at the save times ts
        Us = Us[jnp.searchsorted(times[1:], self.ts)]  # (nts, n, n)

        # === save the propagators
        saved = Saved(Us, None, None)
        saved = self.collect_saved(saved, Us[-1])
        return self.result(saved)
