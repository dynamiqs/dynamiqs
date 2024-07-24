from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from ...result import Saved
from ...utils.operators import eye
from ..apis.sesolve import sesolve
from ..core.abstract_integrator import SEPropagatorIntegrator


class SEPropagatorDiffraxIntegrator(SEPropagatorIntegrator):
    def run(self) -> PyTree:
        initial_states = eye(self.H.shape[-1])[..., None]
        options = eqx.tree_at(
            lambda x: x.cartesian_batching,
            self.options,
            True,
            is_leaf=lambda x: x is None,
        )
        seresult = sesolve(
            self.H,
            initial_states,
            self.ts,
            solver=self.solver,
            gradient=self.gradient,
            options=options,
        )
        saved = self.collect_saved(seresult.states)
        return self.result(saved, seresult.infos)

    def collect_saved(self, states: Array) -> Saved:
        """States is the output of sesolve, which we need to reshape
        into the propagator.
        """
        if self.options.save_states:
            # indices are ...i, t, j. Want to permute them to
            # t, j, i such that the t index is first and each
            # column of the propogator corresponds to each initial state
            ndim = len(states.shape) - 1
            perm = [*list(range(ndim - 3)), ndim - 2, ndim - 1, ndim - 3]
            propagators = jnp.transpose(states[..., 0], perm)
        else:
            # otherwise, sesolve has only saved the final states
            # so we only need to permute the final two axes
            propagators = states[..., 0].swapaxes(-1, -2)
        return Saved(propagators, None, None)
