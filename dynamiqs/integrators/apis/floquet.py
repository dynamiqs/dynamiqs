from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._checks import check_shape
from ...gradient import Gradient
from ...integrators.floquet.floquet_integrator import (
    FloquetIntegrator_t,
    FloquetIntegrator_t0,
)
from ...options import Options
from ...result import FloquetResult, Saved
from ...solver import Solver, Tsit5
from ...time_array import TimeArray
from .._utils import _astimearray, catch_xla_runtime_error, multi_vmap

__all__ = ['floquet']


def floquet(
    H: ArrayLike | TimeArray,
    T: float | ArrayLike,
    tsave: ArrayLike,
    *,
    solver: Solver = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
    safe: bool = False,
) -> FloquetResult:
    r"""Compute Floquet modes $\Phi_{m}(t)$ and quasienergies $\epsilon_m$.

    For a periodically driven system, the Floquet modes $\Phi_{m}(t_0)$ and
    quasienergies $\epsilon_m$ are defined by the eigenvalue equation
    $$
        U(t_0, t_0+T)\Phi_{m}(t_0) = \exp(-i \epsilon_{m} T)\Phi_{m}(t_0),
    $$
    where $U(t_0, t_0+T)$ is the propagator from time t_0 to time $t_0+T$, and $T$ is
    the period of the drive. Typically $t_0$ is taken to be $0$, however that does not
    not always have to be the case. We thus obtain the $\Phi_{m}(t_0)$ and $\epsilon_m$
    by diagonalizing the propagator $U(t_0, t_0+T)$.

    The Floquet modes $\Phi_{m}(t)$ at times $t\neq t_0$ are obtained from the Floquet
    modes $\Phi_{m}(t_0)$ via
    $$
        \Phi_{m}(t) = \exp(i\epsilon_{m}t)U(t_0, t_0+t)\Phi_{m}(t_0).
    $$

    Args:
        H _(array-like or time-array of shape (...H, n, n))_: Hamiltonian.
        T _(array-like of shape (...H))_: Period of the drive. T should have the same
            shape as ...H or should be broadcastable to that shape. This is to allow
            batching over Hamiltonians with differing drive frequencies.
        tsave _(array-like of shape (ntsave,) or (...H, ntsave))_: Times at which to
            compute floquet modes. `tsave` is allowed to have batch dimensions to allow
            for cases where `H` is batched over different drive frequencies. In this
            case, it makes sense to ask for the Floquet modes at different times for
            different batch dimensions.
        solver: Solver for the integration.
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].
        safe: Whether or not to check if the Hamiltonian is actually periodic with the
            supplied period T

    Returns:
        [`dq.FloquetResult`][dynamiqs.FloquetResult] object holding the result of the
            Floquet computation. Use the attribute `floquet_modes` to access the saved
            Floquet modes, and the attribute `quasienergies` the associated quasi
            energies, more details in [`dq.FloquetResult`][dynamiqs.FloquetResult].
    """
    # === convert arguments
    H = _astimearray(H)
    T = jnp.asarray(T)
    tsave = jnp.asarray(tsave)
    # TODO check_times for tsave but for now we are allowing it to be multidimensional

    # === broadcast arguments
    # Different batch Hamiltonians may be periodic with varying periods, so T must be
    # broadcastable to the same shape as H.
    H, T, broadcast_shape = _broadcast_floquet_args(H, T)
    tsave = jnp.broadcast_to(tsave, broadcast_shape + tsave.shape[-1:])

    # === check arguments
    _check_floquet_args(H, T, safe=safe)

    # We implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    return _vectorized_floquet(H, T, tsave, solver, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vectorized_floquet(
    H: TimeArray,
    T: Array,
    tsave: Array,
    solver: Solver,
    gradient: Gradient,
    options: Options,
) -> FloquetResult:
    # We first compute the `t=t_0` Floquet modes
    floquet_result_t0 = _vectorized_floquet_t0(
        H, T, tsave[..., 0], solver, gradient, options
    )
    # We then use these modes to compute the Floquet modes at other times if asked for
    if tsave.shape[-1] > 1:
        return _vectorized_floquet_t(
            H=H,
            T=T,
            tsave=tsave,
            floquet_modes_t0=floquet_result_t0.floquet_modes,
            quasienergies=floquet_result_t0.quasienergies,
            solver=solver,
            gradient=gradient,
            options=options,
        )
    return floquet_result_t0


def _vectorized_floquet_t0(
    H: TimeArray,
    T: Array,
    t0: Array,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> FloquetResult:
    # We flat vectorize over H, T and t0.
    in_axes = (H.in_axes, 0, 0, None, None, None)
    # the result is vectorized over `t0`, `_saved`, `infos` and T
    out_axes = FloquetResult(0, None, None, None, 0, 0, 0)
    nvmap = len(T.shape)
    # vectorize the function
    f = multi_vmap(_floquet_t0, in_axes, out_axes, nvmap)

    return f(H, T, t0, solver, gradient, options)


def _floquet_t0(
    H: TimeArray,
    T: Array,
    t0: Array,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> FloquetResult:
    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === integrator class is always FloquetIntegrator_t0
    integrator = FloquetIntegrator_t0(
        ts=t0, y0=None, H=H, solver=solver, gradient=gradient, options=options, T=T
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _vectorized_floquet_t(
    H: TimeArray,
    T: Array,
    tsave: Array,
    floquet_modes_t0: Array,
    quasienergies: Array,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> FloquetResult:
    # We flat vectorize over H, T, tsave, floquet_modes_t0 and quasienergies
    in_axes = (H.in_axes, 0, 0, 0, 0, None, None, None)
    # the result is vectorized over `tsave`, `_saved`, `infos` and T
    out_axes = FloquetResult(0, None, None, None, 0, 0, 0)
    nvmap = len(T.shape)
    # vectorize the function
    f = multi_vmap(_floquet_t, in_axes, out_axes, nvmap)

    return f(H, T, tsave, floquet_modes_t0, quasienergies, solver, gradient, options)


def _floquet_t(
    H: TimeArray,
    T: Array,
    tsave: Array,
    floquet_modes_t0: Array,
    quasienergies: Array,
    solver: Solver,
    gradient: Gradient,
    options: Options,
) -> FloquetResult:
    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === integrator class is always FloquetIntegrator_t
    integrator = FloquetIntegrator_t(
        ts=tsave,
        y0=None,
        H=H,
        solver=solver,
        gradient=gradient,
        options=options,
        T=T,
        floquet_modes_t0=floquet_modes_t0,
        quasienergies=quasienergies,
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _broadcast_floquet_args(H: TimeArray, T: Array) -> [Array, Array, Array]:
    broadcast_shape = jnp.broadcast_shapes(H.shape[:-2], T.shape)
    H = H.broadcast_to(*(broadcast_shape + H.shape[-2:]))
    T = jnp.broadcast_to(T, broadcast_shape)
    return H, T, broadcast_shape


def _check_floquet_args(H: TimeArray, T: Array, safe: bool = False):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check that the Hamiltonian is periodic with the supplied period
    if safe:
        in_axes = (H.in_axes, 0)
        out_axes = Saved(0, None)
        nvmap = len(T.shape)
        H_0 = multi_vmap(lambda _H, _T: Saved(_H(0), None), in_axes, out_axes, nvmap)
        H_T = multi_vmap(lambda _H, _T: Saved(_H(_T), None), in_axes, out_axes, nvmap)
        periodic = jnp.allclose(H_0(H, T).ysave, H_T(H, T).ysave)
        if not periodic:
            raise ValueError(
                'The Hamiltonian H is not periodic with the supplied period T'
            )
