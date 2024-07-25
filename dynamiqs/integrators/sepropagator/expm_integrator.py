from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from ..._utils import _concatenate_sort
from ...result import Saved
from ...utils.operators import eye
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
        # === find times at which the Hamiltonian changes (between t0 and tfinal)
        t0 = jnp.asarray(self.t0).reshape(-1)
        times = _concatenate_sort(t0, self.H.discontinuity_ts, self.ts)
        inbounds = (times[:-1] >= t0) & (times[1:] <= self.ts[-1])
        t_diffs = jnp.where(inbounds, jnp.diff(times), 0.0)

        # === compute the propagators
        # don't need the last time in times since the hamiltonian is guaranteed
        # to be constant over the region times[-2] to times[-1]
        Hs = jnp.stack([self.H(t) for t in times[:-1]])
        dims = tuple(range(-Hs.ndim + 1, 0))
        t_diffs = jnp.expand_dims(t_diffs, dims)
        step_propagators = expm(-1j * t_diffs * Hs)

        # === combine propagators to get the propagator at each requested time
        def _reduce(prev_prop: Array, next_prop: Array) -> Array:
            # notice the ordering of prev_prop and next_prop, want
            # next_prop to be to the left of prev_prop
            total_prop = next_prop @ prev_prop
            return total_prop, total_prop

        eye_broadcast = jnp.broadcast_to(eye(self.H.shape[-1]), self.H.shape)
        _, propagators_for_times = jax.lax.scan(
            _reduce, eye_broadcast, step_propagators
        )

        # === extract the propagators at the correct times
        # the -1 is because the indices of the propagators are defined by t_diffs,
        # not times itself
        t_idxs = jnp.argmin(jnp.abs(times - self.ts[:, None]), axis=1)
        t_idxs = jnp.where(t_idxs > 0, t_idxs - 1, t_idxs)
        propagators = propagators_for_times[t_idxs]
        # note that we can't take the output of scan as final_prop, because
        # it could correspond to a time window of H.times. However
        # the final element of propagators will correspond to the propagator
        # at the final time
        final_prop = propagators[-1]
        propagators = jnp.moveaxis(propagators, 0, -3)

        # === save the propagators
        saved = Saved(propagators, None, None)
        saved = self.collect_saved(saved, final_prop)
        return self.result(saved)
