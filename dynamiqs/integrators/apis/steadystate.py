from __future__ import annotations

import warnings
from functools import partial

import jax
import jax.numpy as jnp

from ..._checks import check_hermitian, check_qarray_is_dense, check_shape
from ...gradient import Gradient
from ...method import Dopri5, Dopri8, Euler, Kvaerno3, Kvaerno5, Method, Tsit5
from ...options import Options, check_options
from ...qarrays.qarray import QArray, QArrayLike
from ...qarrays.utils import asqarray
from ...result import SteadyStateResult
from ...time_qarray import ConstantTimeQArray, TimeQArray
from .._utils import (
    assert_method_supported,
    astimeqarray,
    cartesian_vmap,
    catch_xla_runtime_error,
    multi_vmap,
)
from ..core.diffrax_integrator import (
    steadystate_dopri5_integrator_constructor,
    steadystate_dopri8_integrator_constructor,
    steadystate_euler_integrator_constructor,
    steadystate_kvaerno3_integrator_constructor,
    steadystate_kvaerno5_integrator_constructor,
    steadystate_tsit5_integrator_constructor,
)


def steadystate(
    H: QArrayLike | ConstantTimeQArray,
    jump_ops: list[QArrayLike | ConstantTimeQArray],
    rho0: QArrayLike,
    *,
    exp_ops: list[QArrayLike] | None = None,
    method: Method = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> SteadyStateResult:
    r"""Solve the Lindblad master equation.
    TODO: fill in documentation.
    """
    # === convert arguments
    H = astimeqarray(H)
    Ls = [astimeqarray(L) for L in jump_ops]
    rho0 = asqarray(rho0)
    if exp_ops is not None:
        exp_ops = [asqarray(E) for E in exp_ops] if len(exp_ops) > 0 else None

    # === check arguments
    _check_steadystate_args(H, Ls, rho0, exp_ops)
    check_options(options, 'steadystate')
    options = options.initialise()

    # === convert rho0 to density matrix
    rho0 = rho0.todm()
    rho0 = check_hermitian(rho0, 'rho0')

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to qarrays
    return _vectorized_steadystate(H, Ls, rho0, exp_ops, method, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('gradient', 'options'))
def _vectorized_steadystate(
    H: TimeQArray,
    Ls: list[TimeQArray],
    rho0: QArray,
    exp_ops: list[QArray] | None,
    method: Method,
    gradient: Gradient | None,
    options: Options,
) -> SteadyStateResult:
    # vectorize input over H, Ls and rho0
    in_axes = (H.in_axes, [L.in_axes for L in Ls], 0, None, None, None, None, None)
    out_axes = SteadyStateResult.out_axes()

    if options.cartesian_batching:
        nvmap = (H.ndim - 2, [L.ndim - 2 for L in Ls], rho0.ndim - 2, 0, 0, 0, 0, 0)
        f = cartesian_vmap(_steadystate, in_axes, out_axes, nvmap)
    else:
        bshape = jnp.broadcast_shapes(*[x.shape[:-2] for x in [H, *Ls, rho0]])
        nvmap = len(bshape)
        # broadcast all vectorized input to same shape
        n = H.shape[-1]
        H = H.broadcast_to(*bshape, n, n)
        Ls = [L.broadcast_to(*bshape, n, n) for L in Ls]
        rho0 = rho0.broadcast_to(*bshape, n, n)
        # vectorize the function
        f = multi_vmap(_steadystate, in_axes, out_axes, nvmap)

    return f(H, Ls, rho0, exp_ops, method, gradient, options)


def _steadystate(
    H: TimeQArray,
    Ls: list[TimeQArray],
    rho0: QArray,
    exp_ops: list[QArray] | None,
    method: Method,
    gradient: Gradient | None,
    options: Options,
) -> SteadyStateResult:
    # === select integrator constructor
    integrator_constructors = {
        Euler: steadystate_euler_integrator_constructor,
        Dopri5: steadystate_dopri5_integrator_constructor,
        Dopri8: steadystate_dopri8_integrator_constructor,
        Tsit5: steadystate_tsit5_integrator_constructor,
        Kvaerno3: steadystate_kvaerno3_integrator_constructor,
        Kvaerno5: steadystate_kvaerno5_integrator_constructor,
    }
    assert_method_supported(method, integrator_constructors.keys())
    integrator_constructor = integrator_constructors[type(method)]

    # === check gradient is supported
    method.assert_supports_gradient(gradient)

    # === init integrator
    integrator = integrator_constructor(
        ts=jnp.array([0.0, jnp.inf]),
        y0=rho0,
        method=method,
        gradient=gradient,
        result_class=SteadyStateResult,
        options=options,
        H=H,
        Ls=Ls,
        Es=exp_ops,
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_steadystate_args(
    H: TimeQArray, Ls: list[TimeQArray], rho0: QArray, exp_ops: list[QArray] | None
):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check Ls shape
    for i, L in enumerate(Ls):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})

    if len(Ls) == 0 and rho0.isket():
        warnings.warn(
            'Argument `jump_ops` is an empty list and argument `rho0` is a ket,'
            ' consider using `dq.sesolve()` to solve the Schrödinger equation.',
            stacklevel=2,
        )

    # === check rho0 shape and layout
    check_shape(rho0, 'rho0', '(..., n, 1)', '(..., n, n)', subs={'...': '...rho0'})
    check_qarray_is_dense(rho0, 'rho0')

    # === check exp_ops shape
    if exp_ops is not None:
        for i, E in enumerate(exp_ops):
            check_shape(E, f'exp_ops[{i}]', '(n, n)')
