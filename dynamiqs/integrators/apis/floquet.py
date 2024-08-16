from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._checks import check_shape
from ...gradient import Gradient
from ...integrators.floquet.floquet_integrator import (
    FloquetIntegrator,
    FloquetIntegratort,
)
from ...options import Options
from ...result import FloquetResult, Saved
from ...solver import Solver, Tsit5
from ...time_array import Shape, TimeArray
from .._utils import _astimearray, _flat_vectorize, catch_xla_runtime_error

__all__ = ['floquet', 'floquet_t']


def floquet(
    H: ArrayLike | TimeArray,
    T: ArrayLike,
    *,
    solver: Solver = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
    safe: bool = False,
) -> FloquetResult:
    r"""Compute Floquet modes $\Phi_{m}(0)$ and quasi energies $\epsilon_m$.

    For a periodically driven system, the Floquet modes $\Phi_{m}(0)$ and quasi energies
    $\epsilon_m$ are defined by the eigenvalue equation
    $$
        U(0, T)\Phi_{m}(0) = \exp(-i \epsilon_{m} T)\Phi_{m}(0),
    $$
    where U(0, T) is the propagator from time 0 to time T, and T is the period
    of the drive. We thus obtain the $\Phi_{m}(0)$ and $\epsilon_m$ by diagonalizing
    the propagator U(0, T).

    Args:
        H _(array-like or time-array of shape (...H, n, n))_: Hamiltonian.
        T _(array-like of shape (...H))_: Period of the drive. T should have the same
            shape as ...H or should be broadcastable to that shape. This is to allow
            batching over Hamiltonians with differing drive frequencies
        solver: Solver for the integration.
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].
        safe: Whether or not to check if the Hamiltonian is actually periodic with the
            supplied period T

    Returns:
        [`dq.FloquetResult`][dynamiqs.FloquetResult] object holding the result of the
            Floquet computation. Use the attribute `floquet_modes` to access the saved
            Floquet modes, and the attribute `quasi_energies` the associated quasi
            energies, more details in [`dq.FloquetResult`][dynamiqs.FloquetResult].
    """
    # === convert arguments
    H = _astimearray(H)
    T = jnp.asarray(T)

    # === broadcast arguments
    H, T, _ = _broadcast_floquet_args(H, T)

    # === check arguments
    _check_floquet_args(H, T, safe=safe)

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    return _vectorized_floquet(H, T, solver, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vectorized_floquet(
    H: TimeArray, T: Array, solver: Solver, gradient: Gradient | None, options: Options
) -> FloquetResult:
    # === vectorize function
    # we vectorize over H. Different batch Hamiltonians may be periodic with
    # varying periods, so T must be broadcastable to the same shape as H.

    out_axes = FloquetResult(False, False, False, False, 0, 0)

    n_batch = (H.in_axes, Shape(T.shape), Shape(), Shape(), Shape())
    f = _flat_vectorize(_floquet, n_batch, out_axes)

    # === apply vectorized function
    return f(H, T, solver, gradient, options)


def _floquet(
    H: TimeArray, T: Array, solver: Solver, gradient: Gradient | None, options: Options
) -> FloquetResult:
    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === integrator class is always FloquetIntegrator
    integrator = FloquetIntegrator(
        jnp.asarray([0, T]), None, H, solver, gradient, options, T
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def floquet_t(
    H: ArrayLike | TimeArray,
    T: ArrayLike,
    *,
    tsave: ArrayLike,
    floquet_modes_0: Array | None = None,
    quasi_energies: Array | None = None,
    solver: Solver = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
    safe: bool = False,
) -> FloquetResult:
    r"""Compute Floquet modes $\Phi_{m}(t)$ and quasi energies $\epsilon_m$.

    The Floquet modes $\Phi_{m}(t)$ are obtained via the Floquet modes $\Phi_{m}(0)$ via
    $$
        \Phi_{m}(t) = \exp(i\epsilon_{m}t)U(0, t)\Phi_{m}(0).
    $$

    Args:
        H _(array-like or time-array of shape (...H, n, n))_: Hamiltonian.
        T _(array-like of shape (...H))_: Period of the drive. T should have the same
            shape as ...H or should be broadcastable to that shape. This is to allow
            batching over Hamiltonians with differing drive frequencies.
        tsave _(array-like of shape (ntsave,))_: Times at which to compute floquet modes
        floquet_modes_0 _(array-like of shape (...H, n, n))_: floquet modes at t=0. The
            shape of floquet_modes_0 should be the same as that of H
        quasi_energies _(array-like of shape (...H, n))_: previously obtained quasi
            energies.
        solver: Solver for the integration.
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].
        safe: Whether or not to check if the Hamiltonian is actually periodic with the
            supplied period T

    Returns:
        [`dq.FloquetResult`][dynamiqs.FloquetResult] object holding the result of the
            Floquet computation. Use the attribute `floquet_modes` to access the saved
            Floquet modes, and the attribute `quasi_energies` the associated quasi
            energies, more details in [`dq.FloquetResult`][dynamiqs.FloquetResult].
    """
    # === convert arguments
    H = _astimearray(H)
    T = jnp.asarray(T)
    tsave = jnp.asarray(tsave)
    # TODO check_times for tsave but for now we are allowing it to be multidimensional

    H, T, broadcast_shape = _broadcast_floquet_args(H, T)
    tsave = jnp.broadcast_to(tsave, broadcast_shape + tsave.shape[-1:])

    # === check arguments
    _check_floquet_args(H, T, safe=safe)

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    return _vectorized_floquet_t(
        H, T, tsave, floquet_modes_0, quasi_energies, solver, gradient, options
    )


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vectorized_floquet_t(
    H: TimeArray,
    T: Array,
    tsave: Array,
    floquet_modes_0: Array | None,
    quasi_energies: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> FloquetResult:
    # === vectorize function
    # as in _vectorized_floquet we vectorize over H. Here in addition to flat batching
    # over H and T we batch over tsave and floquet_result_0

    if floquet_modes_0 is not None:
        f_modes_0_batch = Shape(floquet_modes_0.shape[:-2])
        q_energies_batch = Shape(quasi_energies.shape[:-1])
    else:
        f_modes_0_batch = Shape()
        q_energies_batch = Shape()

    out_axes = FloquetResult(0, False, False, False, 0, 0)

    n_batch = (
        H.in_axes,
        Shape(T.shape),
        Shape(tsave.shape[:-1]),
        f_modes_0_batch,
        q_energies_batch,
        Shape(),
        Shape(),
        Shape(),
    )
    f = _flat_vectorize(_floquet_t, n_batch, out_axes)

    # === apply vectorized function
    return f(H, T, tsave, floquet_modes_0, quasi_energies, solver, gradient, options)


def _floquet_t(
    H: TimeArray,
    T: Array,
    tsave: Array,
    floquet_modes_0: Array | None,
    quasi_energies: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> FloquetResult:
    if floquet_modes_0 is None:
        floquet_result_0 = floquet(
            H, T, solver=solver, gradient=gradient, options=options
        )
        floquet_modes_0 = floquet_result_0.floquet_modes
        quasi_energies = floquet_result_0.quasi_energies

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === integrator class is always FloquetIntegratort
    integrator = FloquetIntegratort(
        tsave, None, H, solver, gradient, options, T, floquet_modes_0, quasi_energies
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
        _check_periodic(H, T)


def _check_periodic(H: TimeArray, T: Array):
    # === check that the Hamiltonian is periodic with the supplied period
    n_batch = (H.in_axes, Shape(T.shape))
    out_axes = Saved(0)
    H_0 = _flat_vectorize(lambda _H, _T: Saved(_H(0.0)), n_batch, out_axes)(H, T)
    H_T = _flat_vectorize(lambda _H, _T: Saved(_H(_T)), n_batch, out_axes)(H, T)
    periodic = jnp.allclose(H_0.ysave, H_T.ysave)
    if not periodic:
        raise ValueError('The Hamiltonian H is not periodic with the supplied period T')
