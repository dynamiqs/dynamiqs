from __future__ import annotations

from abc import abstractmethod
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax._custom_types import RealScalarLike
from jax import Array
from jaxtyping import PyTree, Scalar

from dynamiqs._utils import concatenate_sort

from ..._checks import check_hermitian
from ...qarrays.qarray import QArray
from ...result import MESolveResult, Result, Saved, SolveSaved
from ...utils.general import expm
from ...utils.vectorization import slindbladian, unvectorize, vectorize
from .._utils import ispwc
from ..core.abstract_integrator import BaseIntegrator
from .interfaces import AbstractTimeInterface, MEInterface, SEInterface, SolveInterface
from .save_mixin import AbstractSaveMixin, PropagatorSaveMixin, SolveSaveMixin


class ExpmIntegrator(BaseIntegrator, AbstractSaveMixin, AbstractTimeInterface):
    r"""Integrator solving a linear ODE of the form $dX/dt = AX$ by explicitly
    exponentiating the propagator.

    The matrix $A$ of shape (N, N) is a constant or piecewise constant *generator*.
    The *propagator* between time $t0$ and $t1$ is a matrix of shape (N, N) defined by
    $U(t0, t1) = e^{(t0-t1) A}$.

    We solve two different equations:
    - for sesolve/sepropagator, we solve the Schrödinger equation with $N = n$ and
      $A = -iH$ with H the Hamiltonian,
    - for mesolve/mepropagator, we solve the Lindblad master equation, the problem is
      vectorized, $N = n^2$ and $A = \mathcal{L}$ with $\mathcal{L}$ the Liouvillian.

    We compute different objects:
    - for sesolve/mesolve, the state $X$ is an (N, 1) column vector,
    - for sepropagator/mepropagator, the state $X$ is an (N, N) matrix.
    """

    class Infos(eqx.Module):
        nsteps: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                # note: expm methods can make different number of steps between
                # batch elements when batching over PWC objects
                return (
                    f'avg. {self.nsteps.mean():.1f} steps | infos shape'
                    f' {self.nsteps.shape}'
                )
            return f'{self.nsteps} steps'

    @abstractmethod
    def generator(self, t: RealScalarLike) -> QArray:
        pass

    def run(self) -> Result:
        # === find all times at which to stop in [t0, t1]
        # find all times where the solution should be saved (self.ts) or at which the
        # generator changes (self.discontinuity_ts)
        disc_ts = self.discontinuity_ts
        disc_ts = disc_ts.clip(self.t0, self.t1)
        times = concatenate_sort(jnp.asarray([self.t0]), self.ts, disc_ts)  # (ntimes,)

        # === compute time differences (null for times outside [t0, t1])
        delta_ts = jnp.diff(times)  # (ntimes-1,)

        # === combine the propagators $e^{\Delta t A}$ of each time interval
        # On CPU, computing only the propagators that change, one at a time, is much
        # faster than one batched expm over all intervals. On GPU it is slower whenever
        # several propagators are needed: a single expm syncs with the host on its
        # data-dependent branches, which the batched expm evaluates without branching.
        if jax.default_backend() == 'cpu':
            ylast, saved = self._propagate_sequential(times, delta_ts, disc_ts)
        else:
            ylast, saved = self._propagate_batched(times, delta_ts)
        # saved has shape (ntimes-1, N, 1) if y0 has shape (N, 1) -> compute states
        # saved has shape (ntimes-1, N, N) if y0 has shape (N, N) -> compute propagators

        # === save the propagators
        # extract propagators at the save times ts
        t_idxs = jnp.searchsorted(times[1:], self.ts)  # (nts,)
        saved = jax.tree.map(lambda x: x[t_idxs], saved)

        saved = self.postprocess_saved(saved, ylast[None])

        nsteps = (delta_ts != 0).sum()
        return self.result(saved, infos=self.Infos(nsteps))

    def _propagate_batched(self, times: Array, delta_ts: Array) -> tuple[QArray, Saved]:
        # batch-compute the propagators of all intervals
        As = jax.vmap(self.generator)(times[:-1]).asdense()  # (ntimes-1, N, N)
        step_propagators = expm(delta_ts[:, None, None] * As)  # (ntimes-1, N, N)

        def step(carry: QArray, xs: tuple[Array, QArray]) -> tuple[QArray, Saved]:
            # note the ordering x @ carry: we accumulate propagators from the left
            t, x = xs
            x_next = x @ carry
            # the accumulated propagator holds at the interval's right endpoint t
            return x_next, self.save(t, x_next)

        return jax.lax.scan(step, self.y0, (times[1:], step_propagators))

    def _propagate_sequential(
        self, times: Array, delta_ts: Array, disc_ts: Array
    ) -> tuple[QArray, Saved]:
        # compute a propagator only when the generator or the interval length changes:
        # null intervals are skipped, and an interval of the same length (up to the
        # rounding of the times) as the previous one reuses its propagator
        new_generator = jnp.isin(times[:-1], disc_ts)  # (ntimes-1,)
        tol = 4 * jnp.finfo(delta_ts.dtype).eps

        def step(
            carry: tuple[QArray, QArray, Array], xs: tuple[Array, Array, Array, Array]
        ) -> tuple[tuple[QArray, QArray, Array], Saved]:
            y, propagator, last_dt = carry
            t_start, t_end, dt, new_A = xs

            def skip() -> tuple[QArray, QArray, Array]:
                return y, propagator, last_dt

            def reuse() -> tuple[QArray, QArray, Array]:
                # note the ordering x @ y: we accumulate propagators from the left
                return propagator @ y, propagator, last_dt

            def compute() -> tuple[QArray, QArray, Array]:
                x = expm(dt * self.generator(t_start).asdense())  # (N, N)
                return x @ y, x, dt

            scale = jnp.maximum(jnp.abs(t_start), jnp.abs(t_end))
            same = ~new_A & (jnp.abs(dt - last_dt) <= tol * scale)
            branch = jnp.where(dt == 0, 0, jnp.where(same, 1, 2))
            carry = jax.lax.switch(branch, [skip, reuse, compute])
            # the accumulated propagator holds at the interval's right endpoint t_end
            return carry, self.save(t_end, carry[0])

        # placeholder propagator, never used: the first non-null interval computes one
        propagator = 0 * self.generator(self.t0).asdense()  # (N, N)
        carry = (self.y0, propagator, jnp.asarray(-1.0, delta_ts.dtype))
        xs = (times[:-1], times[1:], delta_ts, new_generator)
        (ylast, _, _), saved = jax.lax.scan(step, carry, xs)
        return ylast, saved


class SEExpmIntegrator(ExpmIntegrator, SEInterface):
    """Integrator solving the Schrödinger equation by explicitly exponentiating the
    propagator.
    """

    def __check_init__(self):
        # check that Hamiltonian is constant or pwc, or a sum of constant/pwc
        if not ispwc(self.H):
            raise TypeError(
                'Method `Expm` requires a constant or piecewise constant Hamiltonian.'
            )

    def generator(self, t: RealScalarLike) -> QArray:
        return -1j * self.H(t)  # (n, n)


class SESolveExpmIntegrator(SEExpmIntegrator, SolveSaveMixin, SolveInterface):
    """Integrator computing the time evolution of the Schrödinger equation by
    explicitly exponentiating the propagator.
    """


sesolve_expm_integrator_constructor = SESolveExpmIntegrator


class SEPropagatorExpmIntegrator(SEExpmIntegrator, PropagatorSaveMixin):
    """Integrator computing the propagator of the Lindblad master equation by
    explicitly exponentiating the propagator.
    """


sepropagator_expm_integrator_constructor = SEPropagatorExpmIntegrator


class MEExpmIntegrator(ExpmIntegrator, MEInterface):
    """Integrator solving the Lindblad master equation by explicitly exponentiating the
    propagator.
    """

    def __check_init__(self):
        # check that Hamiltonian is constant or pwc, or a sum of constant/pwc
        if not ispwc(self.H):
            raise TypeError(
                'Method `Expm` requires a constant or piecewise constant Hamiltonian.'
            )

        # check that all jump operators are constant or pwc, or a sum of constant/pwc
        if not all(ispwc(L) for L in self.Ls):
            raise TypeError(
                'Method `Expm` requires constant or piecewise constant jump operators.'
            )

    def generator(self, t: RealScalarLike) -> QArray:
        return slindbladian(self.H(t), self.L(t))  # (n^2, n^2)


class MESolveExpmIntegrator(MEExpmIntegrator, SolveSaveMixin, SolveInterface):
    """Integrator computing the time evolution of the Lindblad master equation by
    explicitly exponentiating the propagator.
    """

    def __post_init__(self):
        # convert y0 to a density matrix
        self.y0 = self.y0.todm()
        self.y0 = check_hermitian(self.y0, 'y0')

        # convert to vectorized form
        self.y0 = vectorize(self.y0)  # (n^2, 1)

    def save(self, t: Scalar, y: PyTree) -> SolveSaved:
        # TODO: implement bexpect for vectorized operators and convert at the end
        # instead of at each step
        y = unvectorize(y)
        return super().save(t, y)


mesolve_expm_integrator_constructor = partial(
    MESolveExpmIntegrator, result_class=MESolveResult
)


class MEPropagatorExpmIntegrator(MEExpmIntegrator, PropagatorSaveMixin):
    """Integrator computing the propagator of the Lindblad master equation by
    explicitly exponentiating the propagator.
    """


mepropagator_expm_integrator_constructor = MEPropagatorExpmIntegrator
