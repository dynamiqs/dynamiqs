from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._checks import check_shape, check_times
from ...gradient import Gradient
from ...integrators.floquet.floquet_integrator import FloquetIntegrator
from ...options import Options
from ...qarrays.qarray import QArrayLike
from ...result import FloquetResult
from ...solver import Solver, Tsit5
from ...time_array import CallableTimeArray, TimeArray
from .._utils import _astimearray, cartesian_vmap, catch_xla_runtime_error

__all__ = ['floquet']


def floquet(
    H: QArrayLike | TimeArray,
    T: float,
    tsave: ArrayLike,
    *,
    solver: Solver = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> FloquetResult:
    r"""Compute Floquet modes and quasienergies of a periodic closed system.

    The Floquet modes $\Phi_{m}(t_0)$ and corresponding quasienergies $\epsilon_m$ are
    defined by the eigenvalue equation
    $$
        U(t_0, t_0+T)\Phi_{m}(t_0) = \exp(-i \epsilon_{m} T)\Phi_{m}(t_0),
    $$
    where $U(t_0, t_0+T)$ is the propagator over a single period $T$, with $t_0$ some
    arbitrary initial time (typically $t_0 = 0$). According to the Floquet theorem,
    these Floquet modes are periodic with period $T$ and form a complete basis for
    the time evolution of the system. The $t=t_0$ Floquet modes are obtained from
    diagonalization of the above propagator, while the $t \geq t_0$ Floquet modes are
    obtained by propagating the $t=t_0$ Floquet modes forward in time, via
    $$
        \Phi_{m}(t) = \exp(i\epsilon_{m}t)U(t_0, t_0+t)\Phi_{m}(t_0).
    $$

    Warning:
        `floquet` is not yet GPU compatible because `jax.numpy.linalg.eig` is only
        implemented on the CPU backend, [see here](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.eig.html).
        However, a GPU implementation of eig is currently in the works, [see here](https://github.com/jax-ml/jax/pull/24663).

    Note-: Batching over drive periods
        The current API does not yet natively support batching over multiple drive
        periods, for instance if you wanted to batch over Hamiltonians with different
        drive frequencies. This however can be achieved straightforwardly with an
        external call to `jax.vmap`, as follows:

        # TODO: fix before merge
        # ```python
        # import jax
        # import jax.numpy as jnp
        # import dynamiqs as dq


        # def single_floquet(omega):
        #     H = dq.modulated(lambda t: jnp.cos(omega * t), dq.sigmax())
        #     T = 2.0 * jnp.pi / omega
        #     tsave = jnp.linspace(0.0, T, 11)
        #     return dq.floquet(H, T, tsave)


        # omegas = jnp.array([0.9, 1.0, 1.1])
        # batched_floquet = jax.vmap(single_floquet)
        # result = batched_floquet(omegas)
        # ```

    Args:
        H _(qarray-like or time-array of shape (...H, n, n))_: Hamiltonian.
        T: Period of the Hamiltonian. If the Hamiltonian is batched, the period should
            be common over all elements in the batch. To batch over different periods,
            wrap the call to `floquet` in a `jax.vmap`, see above.
        tsave _(array-like of shape (ntsave,))_: Times at which to compute floquet
            modes. The specified times should be ordered, strictly ascending, and such
            that `tsave[-1] - tsave[0] <= T`.
        solver: Solver for the integration. Defaults to
            [`dq.solver.Tsit5`][dynamiqs.solver.Tsit5] (supported:
            [`Tsit5`][dynamiqs.solver.Tsit5],
            [`Dopri5`][dynamiqs.solver.Dopri5],
            [`Dopri8`][dynamiqs.solver.Dopri8],
            [`Kvaerno3`][dynamiqs.solver.Kvaerno3],
            [`Kvaerno5`][dynamiqs.solver.Kvaerno5],
            [`Euler`][dynamiqs.solver.Euler]).
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].

    Returns:
        [`dq.FloquetResult`][dynamiqs.FloquetResult] object holding the result of the
            Floquet computation. Use the attribute `modes` to access the saved
            Floquet modes, and the attribute `quasienergies` for the associated
            quasienergies, more details in [`dq.FloquetResult`][dynamiqs.FloquetResult].
    """
    # === convert arguments
    H = _astimearray(H)
    tsave = jnp.asarray(tsave)

    # === check arguments
    tsave = check_times(tsave, 'tsave')
    H, T, tsave = _check_floquet_args(H, T, tsave)

    # We implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    return _vectorized_floquet(H, T, tsave, solver, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vectorized_floquet(
    H: TimeArray,
    T: float,
    tsave: Array,
    solver: Solver,
    gradient: Gradient,
    options: Options,
) -> FloquetResult:
    # vectorize input over H
    in_axes = (H.in_axes, None, None, None, None, None)
    # vectorize output over `_saved` and `infos`
    out_axes = FloquetResult(None, None, None, None, 0, 0, None)

    # cartesian batching only
    nvmap = (H.ndim - 2, 0, 0, 0, 0, 0, 0)
    f = cartesian_vmap(_floquet, in_axes, out_axes, nvmap)

    return f(H, T, tsave, solver, gradient, options)


def _floquet(
    H: TimeArray,
    T: float,
    tsave: Array,
    solver: Solver,
    gradient: Gradient,
    options: Options,
) -> FloquetResult:
    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === integrator class is always FloquetIntegrator
    integrator = FloquetIntegrator(
        ts=tsave, y0=None, H=H, solver=solver, gradient=gradient, options=options, T=T
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_floquet_args(
    H: TimeArray, T: float, tsave: Array
) -> tuple[TimeArray, float, Array]:
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check that tsave[-1] - tsave[0] <= T
    T, tsave = eqx.error_if(
        (T, tsave),
        tsave[-1] - tsave[0] > T,
        'The time interval spanned by tsave should be less than a single period T, '
        'i.e. `tsave[-1] - tsave[0] <= T`.',
    )

    # === check that the Hamiltonian is periodic with the supplied period
    if not isinstance(H, CallableTimeArray):
        # disable the check for CallableTimeArray as equinox cannot find an
        # underlying array to attach the error message to.
        H = eqx.error_if(
            H,
            jnp.logical_not(jnp.isclose(H(0.0)._data, H(T)._data)),  # noqa: SLF001
            'The Hamiltonian H is not periodic with the supplied period T.',
        )

    return H, T, tsave
