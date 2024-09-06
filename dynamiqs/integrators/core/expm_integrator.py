from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from dynamiqs._utils import _concatenate_sort

from ...utils.quantum_utils.general import expm
from ...utils.vectorization import slindbladian
from .._utils import ispwc
from .abstract_integrator import BaseIntegrator, MEIntegrator


class ExpmIntegrator(BaseIntegrator):
    # Base solver to solve a linear ODE of the form $dX/dt = AX$ by computing
    # propagators. The matrix $A$ of shape (N, N) is a constant or piecewise constant
    # *generator*. The *propagator* between time $t0$ and $t1$ is a matrix of shape
    # (N, N) defined by $U(t0, t1) = e^{(t0-t1) A}$.
    #
    # We solve two different equations:
    # - for sesolve/sepropagator, we solve the Schrödinger equation with $N = n$ and
    #   $A = -iH$ with H the Hamiltonian,
    # - for mesolve/mepropagator, we solve the Lindblad master equation, the problem is
    #   vectorized, $N = n^2$ and $A = \mathcal{L}$ with $\mathcal{L}$ the Liouvillian.
    #
    # We compute different objects:
    # - for sesolve/mesolve, the state $X$ is an (N, 1) column vector,
    # - for sepropagator/mepropagator, the state $X$ is an (N, N) matrix.

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

        # check that Hamiltonian is constant or pwc, or a sum of constant/pwc
        if not ispwc(self.H):
            raise TypeError(
                'Solver `Expm` requires a constant or piecewise constant Hamiltonian.'
            )

    def _generator(self, t: float) -> Array:
        raise NotImplementedError

    def run(self) -> PyTree:
        # === find all times at which to stop in [t0, t1]
        # find all times where the solution should be saved (self.ts) or at which the
        # generator changes (self.discontinuity_ts)
        disc_ts = self.discontinuity_ts
        if disc_ts is not None:
            disc_ts = disc_ts.clip(self.t0, self.t1)
        times = _concatenate_sort(jnp.asarray([self.t0]), self.ts, disc_ts)  # (ntimes,)

        # === compute time differences (null for times outside [t0, t1])
        delta_ts = jnp.diff(times)  # (ntimes-1,)

        # === batch-compute the propagators $e^{\Delta t A}$ on each time interval
        As = jax.vmap(self._generator)(times[:-1])  # (ntimes-1, N, N)
        step_propagators = expm(delta_ts[:, None, None] * As)  # (ntimes-1, N, N)

        # === combine the propagators together
        def step(carry: Array, x: Array) -> tuple[Array, Array]:
            # note the ordering x @ carry: we accumulate propagators from the left
            x_next = x @ carry
            return x_next, self.save(x_next)

        ylast, saved = jax.lax.scan(step, self.y0, step_propagators)
        # saved has shape (ntimes-1, N, 1) if y0 has shape (N, 1) -> compute states
        # saved has shape (ntimes-1, N, N) if y0 has shape (N, N) -> compute propagators

        # === save the propagators
        # extract propagators at the save times ts
        t_idxs = jnp.searchsorted(times[1:], self.ts)  # (nts,)
        saved = jax.tree.map(lambda x: x[t_idxs], saved)

        saved = self.postprocess_saved(saved, ylast[None])

        nsteps = (delta_ts != 0).sum()
        return self.result(saved, infos=self.Infos(nsteps))


class SEExpmIntegrator(ExpmIntegrator):
    def _generator(self, t: float) -> Array:
        return -1j * self.H(t)  # (n, n)


class MEExpmIntegrator(ExpmIntegrator, MEIntegrator):
    def __init__(self, *args):
        super().__init__(*args)

        # check that all jump operators are constant or pwc, or a sum of constant/pwc
        if not all(ispwc(L) for L in self.Ls):
            raise TypeError(
                'Solver `Expm` requires constant or piecewise constant jump operators.'
            )

    def _generator(self, t: float) -> Array:
        return slindbladian(self.H(t), [L(t) for L in self.Ls])  # (n^2, n^2)
