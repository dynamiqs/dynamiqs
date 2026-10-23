from __future__ import annotations

from typing import cast

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree

from .gradient import Gradient
from .method import Event, Method
from .options import Options
from .qarrays.qarray import QArray
from .qarrays.utils import asqarray, to_jax
from .utils.general import unit

__all__ = [
    'FloquetResult',
    'MEPropagatorResult',
    'JSSESolveResult',
    'DSSESolveResult',
    'JSMESolveResult',
    'MESolveLowRankResult',
    'MESolveResult',
    'DSMESolveResult',
    'SEPropagatorResult',
    'SESolveResult',
]


def _memory_bytes(x: Array) -> int:
    return x.itemsize * x.size


def _memory_str(x: Array) -> str:
    mem = _memory_bytes(x)
    if mem < 1024**2:
        return f'{mem / 1024:.1f} Kb'
    elif mem < 1024**3:
        return f'{mem / 1024**2:.1f} Mb'
    else:
        return f'{mem / 1024**3:.1f} Gb'


def _array_str(x: Array | QArray | None) -> str | None:
    # TODO: implement _memory_str for `QArray` rather than converting to JAX array
    if x is None:
        return None
    type_name = 'Array' if isinstance(x, Array) else 'QArray'
    x = to_jax(x)
    return f'{type_name} {x.dtype} {tuple(x.shape)} | {_memory_str(x)}'


# the Saved object holds quantities saved during the equation integration
class Saved(eqx.Module):
    ysave: PyTree | None
    extra: PyTree | None


class SolveSaved(Saved):
    Esave: Array | None


class JumpSolveSaved(SolveSaved):
    clicktimes: Array


class DiffusiveSolveSaved(SolveSaved):
    Isave: Array


class PropagatorSaved(Saved):
    pass


class FloquetSaved(Saved):
    quasienergies: Array


class Result(eqx.Module):
    tsave: Array
    method: Method
    gradient: Gradient | None
    options: Options
    _saved: Saved
    infos: PyTree | None

    @property
    def _ysave(self) -> QArray:
        ysave = self._saved.ysave
        return cast(QArray, ysave)

    @property
    def extra(self) -> PyTree | None:
        return self._saved.extra

    def block_until_ready(self) -> Result:
        """Waits until the result is computed, then returns it."""
        _ = self._ysave.block_until_ready()
        return self

    def _str_parts(self) -> dict[str, str | None]:
        return {
            'Method': type(self.method).__name__,
            'Gradient': (
                type(self.gradient).__name__ if self.gradient is not None else None
            ),
            'Infos': self.infos if self.infos is not None else None,
            'Extra': (eqx.tree_pformat(self.extra) if self.extra is not None else None),
        }

    def __str__(self) -> str:
        parts = self._str_parts()

        # remove None values
        parts = {k: v for k, v in parts.items() if v is not None}

        # pad to align colons
        padding = max(len(k) for k in parts) + 1
        parts_str = '\n'.join(f'{k:<{padding}}: {v}' for k, v in parts.items())
        return f'==== {self.__class__.__name__} ====\n' + parts_str

    @classmethod
    def out_axes(cls) -> Result:
        return cls(None, None, None, None, 0, 0)  # ty: ignore[invalid-argument-type]


class SolveResult(Result):
    _saved: SolveSaved

    @property
    def states(self) -> QArray:
        return self._ysave

    @property
    def final_state(self) -> QArray:
        return cast(QArray, self.states[..., -1, :, :])

    @property
    def expects(self) -> Array | None:
        return self._saved.Esave

    def _str_parts(self) -> dict[str, str | None]:
        d = super()._str_parts()
        return d | {
            'States': _array_str(self.states),
            'Expects': _array_str(self.expects),
        }


class PropagatorResult(Result):
    _saved: PropagatorSaved

    @property
    def propagators(self) -> QArray:
        return self._ysave

    @property
    def final_propagator(self) -> QArray:
        return cast(QArray, self.propagators[..., -1, :, :])

    def _str_parts(self) -> dict[str, str | None]:
        d = super()._str_parts()
        return d | {'Propagators': _array_str(self.propagators)}


class FloquetResult(Result):
    r"""Result of the Floquet modes and quasienergies computation.

    Returned by [`dq.floquet()`][dynamiqs.floquet].

    Attributes:
        modes (qarray of shape (..., ntsave, n, n, 1)): Saved Floquet modes, the
            third-to-last axis indexes the modes.
        quasienergies (array of shape (..., n)): Quasienergies of the Floquet modes.
        T (float): Period of the Hamiltonian.
        infos (PyTree or None): Method-dependent information on the resolution.
        tsave (array of shape (ntsave,)): Times for which results were saved.
        method (Method): Method used.
        gradient (Gradient or None): Gradient used.

    Methods:
        block_until_ready(): Waits until the result is computed, then returns it, see
            [`jax.block_until_ready()`](https://docs.jax.dev/en/latest/_autosummary/jax.block_until_ready.html).
    """

    _saved: FloquetSaved
    T: float

    @property
    def modes(self) -> QArray:
        return self._ysave

    @property
    def quasienergies(self) -> Array:
        return self._saved.quasienergies

    def _str_parts(self) -> dict[str, str | None]:
        d = super()._str_parts()
        return d | {
            'Modes': _array_str(self.modes),
            'Quasienergies': _array_str(self.quasienergies),
        }

    @classmethod
    def out_axes(cls) -> FloquetResult:
        return cls(None, None, None, None, 0, 0, None)  # ty: ignore[invalid-argument-type]


class SESolveResult(SolveResult):
    r"""Result of the Schrödinger equation integration.

    Returned by [`dq.sesolve()`][dynamiqs.sesolve].

    Attributes:
        states (qarray of shape (..., nsave, n, 1)): Saved states,
            with `nsave = ntsave`, or `nsave = 1` if `save_states=False`.
        final_state (qarray of shape (..., n, 1)): Saved final state.
        expects (array of shape (..., len(exp_ops), ntsave) or None): Saved
            expectation values, if specified by `exp_ops`.
        extra (PyTree or None): Extra data saved with `save_extra`, if specified.
        infos (PyTree or None): Method-dependent information on the resolution.
        tsave (array of shape (ntsave,)): Times for which results were saved.
        method (Method): Method used.
        gradient (Gradient or None): Gradient used.

    Methods:
        block_until_ready(): Waits until the result is computed, then returns it, see
            [`jax.block_until_ready()`](https://docs.jax.dev/en/latest/_autosummary/jax.block_until_ready.html).
    """


class MESolveResult(SolveResult):
    r"""Result of the Lindblad master equation integration.

    Returned by [`dq.mesolve()`][dynamiqs.mesolve], except with the
    [`LowRank`][dynamiqs.method.LowRank] method which returns a
    [`dq.MESolveLowRankResult`][dynamiqs.MESolveLowRankResult].

    Attributes:
        states (qarray of shape (..., nsave, n, n)): Saved states,
            with `nsave = ntsave`, or `nsave = 1` if `save_states=False`.
        final_state (qarray of shape (..., n, n)): Saved final state.
        expects (array of shape (..., len(exp_ops), ntsave) or None): Saved
            expectation values, if specified by `exp_ops`.
        extra (PyTree or None): Extra data saved with `save_extra`, if specified.
        infos (PyTree or None): Method-dependent information on the resolution.
        tsave (array of shape (ntsave,)): Times for which results were saved.
        method (Method): Method used.
        gradient (Gradient or None): Gradient used.

    Methods:
        block_until_ready(): Waits until the result is computed, then returns it, see
            [`jax.block_until_ready()`](https://docs.jax.dev/en/latest/_autosummary/jax.block_until_ready.html).
    """


class MESolveLowRankResult(MESolveResult):
    r"""Result of the low-rank Lindblad master equation integration.

    Returned by [`dq.mesolve()`][dynamiqs.mesolve] with the
    [`LowRank`][dynamiqs.method.LowRank] method. The density matrices are
    reconstructed from the low-rank factors $m(t)$ as $\rho(t) = m(t) m(t)^\dag$
    each time `states` or `final_state` is accessed.

    Attributes:
        lowrank_states (qarray of shape (..., nsave, n, rank)): Saved low-rank
            factors $m(t)$, with `nsave = ntsave`, or `nsave = 1` if
            `save_states=False`.
        states (qarray of shape (..., nsave, n, n)): Saved density matrices,
            with `nsave = ntsave`, or `nsave = 1` if `save_states=False`.
        final_state (qarray of shape (..., n, n)): Saved final density matrix.
        expects (array of shape (..., len(exp_ops), ntsave) or None): Saved
            expectation values, if specified by `exp_ops`.
        extra (PyTree or None): Extra data saved with `save_extra`, if specified.
        infos (PyTree or None): Method-dependent information on the resolution.
        tsave (array of shape (ntsave,)): Times for which results were saved.
        method (Method): Method used.
        gradient (Gradient or None): Gradient used.

    Methods:
        block_until_ready(): Waits until the result is computed, then returns it, see
            [`jax.block_until_ready()`](https://docs.jax.dev/en/latest/_autosummary/jax.block_until_ready.html).
    """

    @property
    def lowrank_states(self) -> QArray:
        return self._ysave

    @property
    def states(self) -> QArray:
        m = self.lowrank_states.to_jax()
        rho = m @ m.conj().swapaxes(-2, -1)
        return asqarray(rho, dims=self.lowrank_states.dims)

    def _str_parts(self) -> dict[str, str | None]:
        d = super()._str_parts()
        return d | {'Lowrank states': _array_str(self.lowrank_states)}


class SEPropagatorResult(PropagatorResult):
    r"""Result of the Schrödinger equation propagator computation.

    Returned by [`dq.sepropagator()`][dynamiqs.sepropagator].

    Attributes:
        propagators (qarray of shape (..., nsave, n, n)): Saved propagators,
            with `nsave = ntsave`, or `nsave = 1` if `save_propagators=False`.
        final_propagator (qarray of shape (..., n, n)): Saved final propagator.
        extra (PyTree or None): Extra data saved with `save_extra`, if specified.
        infos (PyTree or None): Method-dependent information on the resolution.
        tsave (array of shape (ntsave,)): Times for which results were saved.
        method (Method): Method used.
        gradient (Gradient or None): Gradient used.

    Methods:
        block_until_ready(): Waits until the result is computed, then returns it, see
            [`jax.block_until_ready()`](https://docs.jax.dev/en/latest/_autosummary/jax.block_until_ready.html).
    """


class MEPropagatorResult(PropagatorResult):
    r"""Result of the Lindblad master equation propagator computation.

    Returned by [`dq.mepropagator()`][dynamiqs.mepropagator]. The propagators are
    superoperators acting on vectorized density matrices.

    Attributes:
        propagators (qarray of shape (..., nsave, n^2, n^2)): Saved propagators,
            with `nsave = ntsave`, or `nsave = 1` if `save_propagators=False`.
        final_propagator (qarray of shape (..., n^2, n^2)): Saved final propagator.
        extra (PyTree or None): Extra data saved with `save_extra`, if specified.
        infos (PyTree or None): Method-dependent information on the resolution.
        tsave (array of shape (ntsave,)): Times for which results were saved.
        method (Method): Method used.
        gradient (Gradient or None): Gradient used.

    Methods:
        block_until_ready(): Waits until the result is computed, then returns it, see
            [`jax.block_until_ready()`](https://docs.jax.dev/en/latest/_autosummary/jax.block_until_ready.html).
    """


class StochasticSolveResult(SolveResult):
    keys: PRNGKeyArray

    @classmethod
    def out_axes(cls) -> StochasticSolveResult:
        return cls(None, None, None, None, 0, 0, 0)  # ty: ignore[invalid-argument-type]

    def mean_states(self) -> QArray:
        """Returns the density matrices averaged over the trajectories."""
        return cast(QArray, self.states.todm().mean(axis=-4))

    def mean_expects(self) -> Array | None:
        """Returns the expectation values averaged over the trajectories."""
        if self.expects is None:
            return None
        return self.expects.mean(axis=-3)


class JumpSolveResult(StochasticSolveResult):
    _saved: JumpSolveSaved

    @property
    def clicktimes(self) -> Array:
        return self._saved.clicktimes

    @property
    def nclicks(self) -> Array:
        return jnp.count_nonzero(~jnp.isnan(self.clicktimes), axis=-1)

    def _str_parts(self) -> dict[str, str | None]:
        d = super()._str_parts()
        return d | {'Clicktimes': _array_str(self.clicktimes)}

    def mean_states(self) -> QArray:
        mean_states = super().mean_states()

        if isinstance(self.method, Event) and self.method.smart_sampling:
            assert self.infos is not None
            noclick_prob = self.infos.noclick_prob[..., None, None, None]
            return unit(
                noclick_prob * self.infos.noclick_states.todm()
                + (1 - noclick_prob) * mean_states,
                psd=True,
            )
        else:
            return mean_states

    def mean_expects(self) -> Array | None:
        if self.expects is None:
            return None

        mean_expect = super().mean_expects()

        if isinstance(self.method, Event) and self.method.smart_sampling:
            assert self.infos is not None
            noclick_prob = self.infos.noclick_prob[..., None, None]
            return (
                noclick_prob * self.infos.noclick_expects
                + (1 - noclick_prob) * mean_expect
            )
        else:
            return mean_expect


class JSSESolveResult(JumpSolveResult):
    r"""Result of the jump stochastic Schrödinger equation integration.

    Returned by [`dq.jssesolve()`][dynamiqs.jssesolve]. For the shape indications we
    define `ntrajs` as the number of trajectories (`ntrajs = len(keys)`).

    Attributes:
        states (qarray of shape (..., ntrajs, nsave, n, 1)): Saved states,
            with `nsave = ntsave`, or `nsave = 1` if `save_states=False`.
        final_state (qarray of shape (..., ntrajs, n, 1)): Saved final state.
        expects (array of shape (..., ntrajs, len(exp_ops), ntsave) or None): Saved
            expectation values, if specified by `exp_ops`.
        clicktimes (array of shape (..., ntrajs, len(jump_ops), nmaxclick)): Times at
            which the detectors clicked. Variable-length array padded with `jnp.nan`
            up to `nmaxclick`.
        nclicks (array of shape (..., ntrajs, len(jump_ops))): Number of clicks of each
            detector.
        keys (PRNG key array of shape (..., ntrajs)): PRNG keys used to sample the
            point processes.
        extra (PyTree or None): Extra data saved with `save_extra`, if specified.
        infos (PyTree or None): Method-dependent information on the resolution. With
            the [`Event`][dynamiqs.method.Event] method and `smart_sampling=True`,
            it holds the no-click trajectory, see
            [`dq.method.Event`][dynamiqs.method.Event].
        tsave (array of shape (ntsave,)): Times for which results were saved.
        method (Method): Method used.
        gradient (Gradient or None): Gradient used.

    Methods:
        mean_states(): Returns the density matrices averaged over the trajectories,
            as a qarray of shape (..., nsave, n, n). With `smart_sampling=True`, the
            no-click trajectory is included with its probability.
        mean_expects(): Returns the expectation values averaged over the trajectories,
            as an array of shape (..., len(exp_ops), ntsave), or `None` if
            `exp_ops` is not specified. With `smart_sampling=True`, the no-click
            trajectory is included with its probability.
        block_until_ready(): Waits until the result is computed, then returns it, see
            [`jax.block_until_ready()`](https://docs.jax.dev/en/latest/_autosummary/jax.block_until_ready.html).
    """


class JSMESolveResult(JumpSolveResult):
    r"""Result of the jump stochastic master equation integration.

    Returned by [`dq.jsmesolve()`][dynamiqs.jsmesolve]. For the shape indications we
    define `ntrajs` as the number of trajectories (`ntrajs = len(keys)`) and `nLm` as
    the number of measured loss channels (those for which the measurement efficiency
    is not zero).

    Attributes:
        states (qarray of shape (..., ntrajs, nsave, n, n)): Saved states,
            with `nsave = ntsave`, or `nsave = 1` if `save_states=False`.
        final_state (qarray of shape (..., ntrajs, n, n)): Saved final state.
        expects (array of shape (..., ntrajs, len(exp_ops), ntsave) or None): Saved
            expectation values, if specified by `exp_ops`.
        clicktimes (array of shape (..., ntrajs, nLm, nmaxclick)): Times at
            which the detectors clicked. Variable-length array padded with `jnp.nan`
            up to `nmaxclick`.
        nclicks (array of shape (..., ntrajs, nLm)): Number of clicks of each
            detector.
        keys (PRNG key array of shape (..., ntrajs)): PRNG keys used to sample the
            point processes.
        extra (PyTree or None): Extra data saved with `save_extra`, if specified.
        infos (PyTree or None): Method-dependent information on the resolution.
        tsave (array of shape (ntsave,)): Times for which results were saved.
        method (Method): Method used.
        gradient (Gradient or None): Gradient used.

    Methods:
        mean_states(): Returns the density matrices averaged over the trajectories,
            as a qarray of shape (..., nsave, n, n).
        mean_expects(): Returns the expectation values averaged over the trajectories,
            as an array of shape (..., len(exp_ops), ntsave), or `None` if
            `exp_ops` is not specified.
        block_until_ready(): Waits until the result is computed, then returns it, see
            [`jax.block_until_ready()`](https://docs.jax.dev/en/latest/_autosummary/jax.block_until_ready.html).
    """


class DiffusiveSolveResult(StochasticSolveResult):
    _saved: DiffusiveSolveSaved

    @property
    def measurements(self) -> Array:
        return self._saved.Isave

    def _str_parts(self) -> dict[str, str | None]:
        d = super()._str_parts()
        return d | {'Measurements': _array_str(self.measurements)}


class DSSESolveResult(DiffusiveSolveResult):
    r"""Result of the diffusive stochastic Schrödinger equation integration.

    Returned by [`dq.dssesolve()`][dynamiqs.dssesolve]. For the shape indications we
    define `ntrajs` as the number of trajectories (`ntrajs = len(keys)`).

    Attributes:
        states (qarray of shape (..., ntrajs, nsave, n, 1)): Saved states,
            with `nsave = ntsave`, or `nsave = 1` if `save_states=False`.
        final_state (qarray of shape (..., ntrajs, n, 1)): Saved final state.
        expects (array of shape (..., ntrajs, len(exp_ops), ntsave) or None): Saved
            expectation values, if specified by `exp_ops`.
        measurements (array of shape (..., ntrajs, len(jump_ops), ntsave-1)): Saved
            measurements, time-averaged over each interval between two
            consecutive times in `tsave`.
        keys (PRNG key array of shape (..., ntrajs)): PRNG keys used to sample the
            Wiener processes.
        extra (PyTree or None): Extra data saved with `save_extra`, if specified.
        infos (PyTree or None): Method-dependent information on the resolution.
        tsave (array of shape (ntsave,)): Times for which results were saved.
        method (Method): Method used.
        gradient (Gradient or None): Gradient used.

    Methods:
        mean_states(): Returns the density matrices averaged over the trajectories,
            as a qarray of shape (..., nsave, n, n).
        mean_expects(): Returns the expectation values averaged over the trajectories,
            as an array of shape (..., len(exp_ops), ntsave), or `None` if
            `exp_ops` is not specified.
        block_until_ready(): Waits until the result is computed, then returns it, see
            [`jax.block_until_ready()`](https://docs.jax.dev/en/latest/_autosummary/jax.block_until_ready.html).
    """


class DSMESolveResult(DiffusiveSolveResult):
    r"""Result of the diffusive stochastic master equation integration.

    Returned by [`dq.dsmesolve()`][dynamiqs.dsmesolve]. For the shape indications we
    define `ntrajs` as the number of trajectories (`ntrajs = len(keys)`) and `nLm` as
    the number of measured loss channels (those for which the measurement efficiency
    is not zero).

    Attributes:
        states (qarray of shape (..., ntrajs, nsave, n, n)): Saved states,
            with `nsave = ntsave`, or `nsave = 1` if `save_states=False`.
        final_state (qarray of shape (..., ntrajs, n, n)): Saved final state.
        expects (array of shape (..., ntrajs, len(exp_ops), ntsave) or None): Saved
            expectation values, if specified by `exp_ops`.
        measurements (array of shape (..., ntrajs, nLm, ntsave-1)): Saved
            measurements, time-averaged over each interval between two
            consecutive times in `tsave`.
        keys (PRNG key array of shape (..., ntrajs)): PRNG keys used to sample the
            Wiener processes.
        extra (PyTree or None): Extra data saved with `save_extra`, if specified.
        infos (PyTree or None): Method-dependent information on the resolution.
        tsave (array of shape (ntsave,)): Times for which results were saved.
        method (Method): Method used.
        gradient (Gradient or None): Gradient used.

    Methods:
        mean_states(): Returns the density matrices averaged over the trajectories,
            as a qarray of shape (..., nsave, n, n).
        mean_expects(): Returns the expectation values averaged over the trajectories,
            as an array of shape (..., len(exp_ops), ntsave), or `None` if
            `exp_ops` is not specified.
        block_until_ready(): Waits until the result is computed, then returns it, see
            [`jax.block_until_ready()`](https://docs.jax.dev/en/latest/_autosummary/jax.block_until_ready.html).
    """
