"""Benchmark case registry.

Cases come in two tiers (see `Tier`):

- `physics` -- representative user workloads, one per solver API, run on every change to
  catch regressions with `python -m benchmarks compare`;
- `features` -- families of cases that differ in exactly one knob (layout, method,
  option, gradient), registered consecutively so that two adjacent rows of a single run
  answer a question on their own ("is SparseDIA faster than dense here?").

Feature families compare methods at equal *tolerance*, not equal accuracy: this suite
times only, correctness is the job of `tests/`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

import dynamiqs as dq
from dynamiqs.gradient import Gradient
from dynamiqs.method import Method
from dynamiqs.qarrays.layout import Layout

from . import systems


class Tier(Enum):
    PHYSICS = 'physics'  # representative user workloads, run on every change
    FEATURES = 'features'  # one-knob comparison families, run on demand

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Case:
    """A single benchmark case.

    Attributes:
        name: Benchmark family name, e.g. `'mesolve_cavity'`.
        params: Case parameters, e.g. `{'n': 128, 'batch': 100}`. Together with `name`
            they identify a case across runs (used to align rows in `compare`), so they
            must stay JSON-serializable -- variant knobs are recorded as strings, e.g.
            `{'layout': 'dia'}`.
        build: Zero-argument setup function (not timed) returning the zero-argument
            run closure. The runner jits the closure, times its ahead-of-time
            compilation, then times its execution. The closure must return a JAX
            pytree, which the runner blocks on with `jax.block_until_ready()`.
        tier: Tier the case belongs to.
        jit: If `False`, the runner calls the closure directly instead of wrapping it in
            `jax.jit`. The fixed-step stochastic integrators need `tsave` to stay
            concrete (they derive the number of steps from it in Python), so they cannot
            be traced by an outer jit; they compile themselves on their first call.
    """

    name: str
    params: dict[str, Any]
    build: Callable[[], Callable[[], Any]]
    tier: Tier = field(default=Tier.PHYSICS)
    jit: bool = field(default=True)

    @property
    def key(self) -> str:
        params = ','.join(f'{k}={v}' for k, v in sorted(self.params.items()))
        return f'{self.name}[{params}]'


# ======================================================================================
# run functions
# ======================================================================================


def _sesolve_transmon(n: int) -> Callable[[], Any]:
    H, psi0, tsave = systems.transmon(n)
    return lambda: dq.sesolve(H, psi0, tsave, progress_meter=False)


def _sesolve_spin_chain(nspin: int, layout: Layout = dq.dia) -> Callable[[], Any]:
    H, psi0, tsave = systems.spin_chain(nspin, layout=layout)
    return lambda: dq.sesolve(H, psi0, tsave, progress_meter=False)


def _sesolve_pwc(n: int, nseg: int) -> Callable[[], Any]:
    H, psi0, tsave = systems.pwc_drive(n, nseg)
    return lambda: dq.sesolve(H, psi0, tsave, progress_meter=False)


def _sesolve_cavity(
    n: int, batch: int = 1, method: Method | None = None
) -> Callable[[], Any]:
    H, psi0, tsave = systems.cavity_closed(n, batch=batch)
    method = dq.method.Tsit5() if method is None else method
    return lambda: dq.sesolve(H, psi0, tsave, method=method, progress_meter=False)


def _mesolve_cavity(
    n: int,
    *,
    layout: Layout = dq.dia,
    method: Method | None = None,
    ntsave: int | None = None,
    save_states: bool = True,
    vectorized: bool = False,
    assume_hermitian: bool = True,
) -> Callable[[], Any]:
    H, Ls, rho0, tsave = systems.cavity(n, layout=layout)
    if ntsave is not None:
        tsave = jnp.linspace(tsave[0], tsave[-1], ntsave)
    method = dq.method.Tsit5() if method is None else method
    exp_ops = [dq.number(n, layout=layout)]

    def run() -> dq.MESolveResult:
        return dq.mesolve(
            H,
            Ls,
            rho0,
            tsave,
            exp_ops=exp_ops,
            method=method,
            save_states=save_states,
            vectorized=vectorized,
            assume_hermitian=assume_hermitian,
            progress_meter=False,
        )

    return run


def _mesolve_cat(
    n: int, alpha: float, batch: int, method: Method | None = None
) -> Callable[[], Any]:
    H, Ls, rho0, tsave = systems.cat(n, alpha, batch=batch)
    method = dq.method.Tsit5() if method is None else method
    return lambda: dq.mesolve(H, Ls, rho0, tsave, method=method, progress_meter=False)


def _mesolve_cross_resonance(n: int, batch: int) -> Callable[[], Any]:
    H, Ls, rho0, tsave = systems.cross_resonance(n, batch=batch)
    return lambda: dq.mesolve(H, Ls, rho0, tsave, progress_meter=False)


def _mesolve_grad(
    n: int, nparams: int, gradient: Gradient | None = None
) -> Callable[[], Any]:
    # pulse optimization: gradient of a scalar loss with respect to the `nparams`
    # amplitudes of a piecewise-constant drive
    gradient = dq.gradient.BackwardCheckpointed() if gradient is None else gradient
    a = dq.destroy(n)
    number_op = dq.number(n)
    Ls = [jnp.sqrt(1.0) * a]
    rho0 = dq.coherent_dm(n, 1.0)
    tgate = 5.0
    tsave = jnp.linspace(0.0, tgate, 101)
    segments = jnp.linspace(0.0, tgate, nparams + 1)
    drive = a + a.dag()

    def loss(eps: Array) -> Array:
        H = 1.0 * number_op + dq.pwc(segments, eps, drive)
        result = dq.mesolve(H, Ls, rho0, tsave, gradient=gradient, progress_meter=False)
        return dq.expect(number_op, result.final_state).real

    # `Forward` cannot be reverse-mode differentiated: it needs `jax.jacfwd`, which for
    # a scalar loss returns the same gradient as `jax.grad`
    grad = jax.jacfwd if isinstance(gradient, dq.gradient.Forward) else jax.grad
    eps0 = 0.5 * jnp.ones(nparams)
    return lambda: grad(loss)(eps0)


def _propagator_tsave(tsave: Array, ntsave: int) -> Array:
    # `Expm` exponentiates the generator once per interval, so the cost is set by the
    # number of save times: keep it well below the 101 points of the solve cases
    return jnp.linspace(tsave[0], tsave[-1], ntsave)


def _sepropagator_expm(n: int, ntsave: int) -> Callable[[], Any]:
    H, _, _, tsave = systems.cavity(n)
    tsave = _propagator_tsave(tsave, ntsave)
    return lambda: dq.sepropagator(
        H, tsave, method=dq.method.Expm(), progress_meter=False
    )


def _mepropagator_expm(n: int, ntsave: int) -> Callable[[], Any]:
    H, Ls, _, tsave = systems.cavity(n)
    tsave = _propagator_tsave(tsave, ntsave)
    return lambda: dq.mepropagator(
        H, Ls, tsave, method=dq.method.Expm(), progress_meter=False
    )


def _floquet_driven_kerr(n: int) -> Callable[[], Any]:
    H, period, tsave = systems.driven_kerr(n)
    return lambda: dq.floquet(H, period, tsave, progress_meter=False)


def _sde_ingredients(n: int, ntraj: int) -> tuple[Any, list[Any], Any, Any, Array, Any]:
    # the fixed-step stochastic integrators assume `tsave` is linearly spaced with each
    # value an exact multiple of `dt`, so keep `tsave` on a round grid
    H, Ls, rho0, _ = systems.cavity(n)
    tsave = jnp.linspace(0.0, 1.0, 11)
    keys = jax.random.split(jax.random.key(42), ntraj)
    return H, Ls, dq.coherent(n, 1.0), rho0, tsave, keys


def _jssesolve_cavity(n: int, ntraj: int, dt: float) -> Callable[[], Any]:
    H, Ls, psi0, _, tsave, keys = _sde_ingredients(n, ntraj)
    method = dq.method.EulerJump(dt=dt)
    return lambda: dq.jssesolve(H, Ls, psi0, tsave, keys, method=method)


def _jsmesolve_cavity(n: int, ntraj: int, dt: float) -> Callable[[], Any]:
    H, Ls, _, rho0, tsave, keys = _sde_ingredients(n, ntraj)
    thetas, etas = jnp.zeros(len(Ls)), 0.8 * jnp.ones(len(Ls))
    method = dq.method.EulerJump(dt=dt)
    return lambda: dq.jsmesolve(H, Ls, thetas, etas, rho0, tsave, keys, method=method)


def _dssesolve_cavity(n: int, ntraj: int, dt: float) -> Callable[[], Any]:
    H, Ls, psi0, _, tsave, keys = _sde_ingredients(n, ntraj)
    method = dq.method.EulerMaruyama(dt=dt)
    return lambda: dq.dssesolve(H, Ls, psi0, tsave, keys, method=method)


def _dsmesolve_cavity(n: int, ntraj: int, dt: float) -> Callable[[], Any]:
    H, Ls, _, rho0, tsave, keys = _sde_ingredients(n, ntraj)
    etas = 0.8 * jnp.ones(len(Ls))
    method = dq.method.Rouchon1(dt=dt)
    return lambda: dq.dsmesolve(H, Ls, etas, rho0, tsave, keys, method=method)


# ======================================================================================
# parameter grids
# ======================================================================================

# problem sizes, tuned so that each tier fits its wall-clock budget on a laptop CPU
_FULL_GRID = {
    # physics tier
    'transmon': [12],  # (n,)
    'spin_chain': [4, 8, 12],  # (nspin,)
    'pwc': [(128, 100), (1024, 100)],  # (n, nseg)
    'cavity': [64, 256],  # (n,)
    'cat': [(32, 2.0, 1), (48, 3.0, 8)],  # (n, alpha, batch)
    'cross_resonance': [(3, 16)],  # (n, batch)
    'grad': [(32, 20)],  # (n, nparams)
    'sepropagator': [(64, 21)],  # (n, ntsave)
    'mepropagator': [(16, 21)],  # (n, ntsave)
    'floquet': [32],  # (n,)
    'sde': [(16, 32, 1e-3)],  # (n, ntraj, dt)
    # features tier
    'feat_layout_mesolve': 128,  # n
    'feat_layout_sesolve': 10,  # nspin
    'feat_rouchon': 64,  # n
    'feat_expm_sesolve': 64,  # n
    'feat_expm_mesolve': 16,  # n
    'feat_vectorized': [16, 32, 64],  # (n,)
    'feat_assume_hermitian': 128,  # n
    'feat_save_states': (256, 501),  # (n, ntsave)
    'feat_gradient': (16, [1, 20]),  # (n, [nparams])
    'feat_lowrank': (64, [4, 8]),  # (n, [rank])
    'feat_batch': (128, [1, 16, 256]),  # (n, [batch])
}

# tiny sizes for CPU CI and sanity runs, one point per family
_QUICK_GRID = {
    'transmon': [6],
    'spin_chain': [4],
    'pwc': [(8, 20)],
    'cavity': [8],
    'cat': [(8, 1.0, 2)],
    'cross_resonance': [(2, 2)],
    'grad': [(8, 4)],
    'sepropagator': [(8, 11)],
    'mepropagator': [(4, 11)],
    'floquet': [8],
    'sde': [(4, 2, 1e-2)],
    'feat_layout_mesolve': 8,
    'feat_layout_sesolve': 4,
    'feat_rouchon': 8,
    'feat_expm_sesolve': 8,
    'feat_expm_mesolve': 4,
    'feat_vectorized': [4],
    'feat_assume_hermitian': 8,
    'feat_save_states': (8, 51),
    'feat_gradient': (4, [1, 2]),
    'feat_lowrank': (8, [2]),
    'feat_batch': (8, [1, 2]),
}

_Grid = dict[str, Any]
_partial = functools.partial


# ======================================================================================
# physics tier: representative user workloads
# ======================================================================================


def _closed_system_cases(g: _Grid) -> list[Case]:
    cases = []

    # single-qubit gate: small Hilbert space, smooth analytic pulse re-evaluated at
    # every step -- the regime dominated by per-step overhead, not matrix products
    for n in g['transmon']:
        build = _partial(_sesolve_transmon, n)
        cases.append(Case('sesolve_transmon', {'n': n}, build))

    # many-body ket dynamics: Hamiltonian assembled from tensor products of Paulis
    for nspin in g['spin_chain']:
        build = _partial(_sesolve_spin_chain, nspin)
        cases.append(Case('sesolve_spin_chain', {'nspin': nspin}, build))

    # piecewise-constant drive: `nrej` reports what the discontinuities cost the
    # adaptive stepper
    for n, nseg in g['pwc']:
        build = _partial(_sesolve_pwc, n, nseg)
        cases.append(Case('sesolve_pwc', {'n': n, 'nseg': nseg}, build))

    return cases


def _open_system_cases(g: _Grid) -> list[Case]:
    cases = []

    # canonical large open bosonic system, constant operators: the n^2 density-matrix
    # scaling
    for n in g['cavity']:
        build = _partial(_mesolve_cavity, n)
        cases.append(Case('mesolve_cavity', {'n': n}, build))

    # cat qubit stabilized by two-photon dissipation, batched over the drive amplitude:
    # the shape of a parameter scan
    for n, alpha, batch in g['cat']:
        build = _partial(_mesolve_cat, n, alpha, batch)
        params = {'n': n, 'alpha': alpha, 'batch': batch}
        cases.append(Case('mesolve_cat', params, build))

    # two-transmon gate calibration sweep: small Hilbert space, several jump operators
    for n, batch in g['cross_resonance']:
        build = _partial(_mesolve_cross_resonance, n, batch)
        params = {'n': n * n, 'batch': batch}
        cases.append(Case('mesolve_cross_resonance', params, build))

    # pulse optimization: reverse-mode gradient of a scalar loss through `mesolve`
    for n, nparams in g['grad']:
        build = _partial(_mesolve_grad, n, nparams)
        cases.append(Case('mesolve_grad', {'n': n, 'nparams': nparams}, build))

    return cases


def _propagator_cases(g: _Grid) -> list[Case]:
    cases = []

    # propagators of a constant generator by explicit matrix exponentiation
    for n, ntsave in g['sepropagator']:
        build = _partial(_sepropagator_expm, n, ntsave)
        cases.append(Case('sepropagator_expm', {'n': n, 'ntsave': ntsave}, build))

    # same for the Liouvillian, kept small because of the O(n^6) scaling
    for n, ntsave in g['mepropagator']:
        build = _partial(_mepropagator_expm, n, ntsave)
        cases.append(Case('mepropagator_expm', {'n': n, 'ntsave': ntsave}, build))

    # Floquet modes: one-period propagator, eigendecomposition, forward propagation
    for n in g['floquet']:
        build = _partial(_floquet_driven_kerr, n)
        cases.append(Case('floquet_driven_kerr', {'n': n}, build))

    return cases


def _stochastic_cases(g: _Grid) -> list[Case]:
    # the four stochastic unravelings, all fixed-step and key-batched over trajectories
    families = [
        ('jssesolve_cavity', _jssesolve_cavity),
        ('jsmesolve_cavity', _jsmesolve_cavity),
        ('dssesolve_cavity', _dssesolve_cavity),
        ('dsmesolve_cavity', _dsmesolve_cavity),
    ]
    return [
        Case(
            name,
            {'n': n, 'ntraj': ntraj, 'dt': dt},
            _partial(builder, n, ntraj, dt),
            jit=False,
        )
        for name, builder in families
        for n, ntraj, dt in g['sde']
    ]


# ======================================================================================
# features tier: single-knob comparisons
# ======================================================================================

_LAYOUTS = {'dia': dq.dia, 'dense': dq.dense}
_TIER = Tier.FEATURES


def _layout_cases(g: _Grid) -> list[Case]:
    # SparseDIA vs dense, on the two operator structures the library sees most: banded
    # bosonic ladder operators, and tensor products of Paulis
    n, nspin = g['feat_layout_mesolve'], g['feat_layout_sesolve']
    return [
        *[
            Case(
                'feat_layout_mesolve',
                {'n': n, 'layout': name},
                _partial(_mesolve_cavity, n, layout=layout),
                _TIER,
            )
            for name, layout in _LAYOUTS.items()
        ],
        *[
            Case(
                'feat_layout_sesolve',
                {'nspin': nspin, 'layout': name},
                _partial(_sesolve_spin_chain, nspin, layout),
                _TIER,
            )
            for name, layout in _LAYOUTS.items()
        ],
    ]


def _method_cases(g: _Grid) -> list[Case]:
    cases = []

    # Lindblad-tailored schemes vs a generic explicit RK, at equal tolerance
    n = g['feat_rouchon']
    rouchon = {
        'Tsit5': dq.method.Tsit5(),
        'Rouchon2': dq.method.Rouchon2(),
        'Rouchon3': dq.method.Rouchon3(),
    }
    for name, method in rouchon.items():
        build = _partial(_mesolve_cavity, n, method=method)
        params = {'n': n, 'method': name}
        cases.append(Case('feat_method_rouchon', params, build, _TIER))

    # explicit matrix exponentiation vs stepping, on a constant generator. `Expm` pays
    # for one exponential per save interval whatever the state, so it is priced against
    # `sepropagator_expm`/`mepropagator_expm`, where that cost buys a full propagator
    expm = {'Tsit5': dq.method.Tsit5(), 'Expm': dq.method.Expm()}
    n = g['feat_expm_sesolve']
    for name, method in expm.items():
        build = _partial(_sesolve_cavity, n, 1, method)
        params = {'n': n, 'method': name}
        cases.append(Case('feat_method_expm_sesolve', params, build, _TIER))
    n = g['feat_expm_mesolve']
    for name, method in expm.items():
        build = _partial(_mesolve_cavity, n, method=method)
        params = {'n': n, 'method': name}
        cases.append(Case('feat_method_expm_mesolve', params, build, _TIER))

    return cases


def _option_cases(g: _Grid) -> list[Case]:
    cases = []

    # vectorized Liouvillian: the advantage shrinks as `n` grows, and the explicit
    # n^2 x n^2 superoperator sets a hard memory ceiling not far above the large point.
    # `assume_hermitian` is ignored when `vectorized=True`, so it is turned off in both
    # variants to isolate a single knob
    for n in g['feat_vectorized']:
        for vectorized in (False, True):
            build = _partial(
                _mesolve_cavity, n, vectorized=vectorized, assume_hermitian=False
            )
            params = {'n': n, 'vectorized': vectorized}
            cases.append(Case('feat_vectorized', params, build, _TIER))

    # evolving only the Hermitian part of rho halves the vector-field matmuls -- this
    # says how much of the total that actually is
    n = g['feat_assume_hermitian']
    for assume_hermitian in (True, False):
        build = _partial(_mesolve_cavity, n, assume_hermitian=assume_hermitian)
        params = {'n': n, 'assume_hermitian': assume_hermitian}
        cases.append(Case('feat_assume_hermitian', params, build, _TIER))

    # cost of materializing and saving states, versus expectation values only. This is
    # memory- and IO-bound, so it barely registers on CPU: the case earns its place on
    # GPU runs. `exp_ops` is set in both variants
    n, ntsave = g['feat_save_states']
    for save_states in (True, False):
        build = _partial(_mesolve_cavity, n, ntsave=ntsave, save_states=save_states)
        params = {'n': n, 'ntsave': ntsave, 'save_states': save_states}
        cases.append(Case('feat_save_states', params, build, _TIER))

    return cases


def _scaling_cases(g: _Grid) -> list[Case]:
    cases = []

    # forward- vs reverse-mode: forward should win at one parameter, reverse at many
    n, nparams_list = g['feat_gradient']
    gradients = {
        'BackwardCheckpointed': dq.gradient.BackwardCheckpointed(),
        'Forward': dq.gradient.Forward(),
    }
    for nparams in nparams_list:
        for name, gradient in gradients.items():
            build = _partial(_mesolve_grad, n, nparams, gradient)
            params = {'n': n, 'nparams': nparams, 'gradient': name}
            cases.append(Case('feat_gradient', params, build, _TIER))

    # rank-truncated evolution vs the full density matrix. The ranks are kept low: the
    # method is sensitive to the time-step error, and a rank too large for the problem
    # makes the adaptive stepper collapse rather than converge
    n, ranks = g['feat_lowrank']
    lowrank = {'Tsit5': dq.method.Tsit5()} | {
        f'LowRank{rank}': dq.method.LowRank(rank=rank, key=jax.random.key(0))
        for rank in ranks
    }
    for name, method in lowrank.items():
        build = _partial(_mesolve_cat, n, 2.0, 1, method)
        cases.append(Case('feat_lowrank', {'n': n, 'method': name}, build, _TIER))

    # marginal cost of a batch element: near-flat on GPU until saturation, near-linear
    # on CPU
    n, batches = g['feat_batch']
    for batch in batches:
        build = _partial(_sesolve_cavity, n, batch)
        cases.append(Case('feat_batch', {'n': n, 'batch': batch}, build, _TIER))

    return cases


# ======================================================================================
# registry
# ======================================================================================


def benchmark_cases(quick: bool = False, tier: Tier | None = None) -> list[Case]:
    """Return the list of benchmark cases.

    Args:
        quick: If `True`, use tiny problem sizes (for CPU CI and sanity runs).
        tier: If given, only return the cases of that tier; `None` returns all of them.
    """
    g = _QUICK_GRID if quick else _FULL_GRID
    cases = [
        *_closed_system_cases(g),
        *_open_system_cases(g),
        *_propagator_cases(g),
        *_stochastic_cases(g),
        *_layout_cases(g),
        *_method_cases(g),
        *_option_cases(g),
        *_scaling_cases(g),
    ]
    return cases if tier is None else [c for c in cases if c.tier is tier]
