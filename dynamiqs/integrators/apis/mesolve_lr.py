from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from ..._checks import check_qarray_is_dense, check_shape, check_times
from ...gradient import Gradient
from ...method import Dopri5, Dopri8, Euler, Kvaerno3, Kvaerno5, Method, Tsit5
from ...options import Options, check_options
from ...qarrays.qarray import QArray, QArrayLike
from ...qarrays.utils import asqarray
from ...result import MESolveLRResult
from ...time_qarray import TimeQArray
from .._utils import (
    assert_method_supported,
    astimeqarray,
    cartesian_vmap,
    catch_xla_runtime_error,
    multi_vmap,
)
from ..core.low_rank_integrator import (
    initialize_m0_from_dm,
    initialize_m0_from_ket,
    mesolve_lr_dopri5_integrator_constructor,
    mesolve_lr_dopri8_integrator_constructor,
    mesolve_lr_euler_integrator_constructor,
    mesolve_lr_kvaerno3_integrator_constructor,
    mesolve_lr_kvaerno5_integrator_constructor,
    mesolve_lr_tsit5_integrator_constructor,
)


def mesolve_lr(
    H: QArrayLike | TimeQArray,
    jump_ops: list[QArrayLike | TimeQArray],
    rho0: QArrayLike,
    tsave: ArrayLike,
    *,
    M: int,
    exp_ops: list[QArrayLike] | None = None,
    method: Method = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
    normalize_each_eval: bool = True,
    gram_reg: float = 0.0,
    eps_init: float | None = None,
    key: PRNGKeyArray | None = None,
) -> MESolveLRResult:
    r"""Solve the Lindblad master equation with a low-rank factorization.

    This solver evolves a factor `m(t)` such that the density matrix is approximated by
    `rho(t) = m(t) m(t)^\dagger`. It follows Dynamiqs conventions for time-dependent
    operators, batching, and options, while using a low-rank internal state. Set
    `options.save_factors_only=True` to save `m(t)` instead of `rho(t)` in
    `result.states`.
    """
    # === convert arguments
    H = astimeqarray(H)
    Ls = [astimeqarray(L) for L in jump_ops]
    rho0 = asqarray(rho0)
    try:
        M = int(M)
    except (TypeError, ValueError) as exc:
        raise TypeError('Argument `M` must be an int.') from exc
    gram_reg = float(gram_reg)
    if eps_init is not None:
        eps_init = float(eps_init)
    tsave = jnp.asarray(tsave)
    if exp_ops is not None:
        exp_ops = [asqarray(E) for E in exp_ops] if len(exp_ops) > 0 else None

    # === check arguments
    _check_mesolve_lr_args(H, Ls, rho0, exp_ops, M, gram_reg, eps_init)
    tsave = check_times(tsave, 'tsave')
    check_options(options, 'mesolve_lr')
    options = options.initialise()

    if key is not None:
        key = jnp.asarray(key)

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to qarrays
    f = _vectorized_mesolve_lr
    f = jax.jit(
        f,
        static_argnames=(
            'gradient',
            'options',
            'M',
            'normalize_each_eval',
            'gram_reg',
            'eps_init',
        ),
    )
    return f(
        H,
        Ls,
        rho0,
        tsave,
        exp_ops,
        method,
        gradient,
        options,
        M,
        normalize_each_eval,
        gram_reg,
        eps_init,
        key,
    )


@catch_xla_runtime_error
def _vectorized_mesolve_lr(
    H: TimeQArray,
    Ls: list[TimeQArray],
    rho0: QArray,
    tsave: Array,
    exp_ops: list[QArray] | None,
    method: Method,
    gradient: Gradient | None,
    options: Options,
    M: int,
    normalize_each_eval: bool,
    gram_reg: float,
    eps_init: float | None,
    key: PRNGKeyArray | None,
) -> MESolveLRResult:
    # vectorize input over H, Ls and rho0
    in_axes = (H.in_axes, [L.in_axes for L in Ls], 0, *(None,) * 10)
    out_axes = MESolveLRResult.out_axes()

    if options.cartesian_batching:
        nvmap = (
            H.ndim - 2,
            [L.ndim - 2 for L in Ls],
            rho0.ndim - 2,
            *(0,) * 10,
        )
        f = cartesian_vmap(_mesolve_lr, in_axes, out_axes, nvmap)
    else:
        bshape = jnp.broadcast_shapes(*[x.shape[:-2] for x in [H, *Ls, rho0]])
        nvmap = len(bshape)
        # broadcast all vectorized input to same shape
        n = H.shape[-1]
        H = H.broadcast_to(*bshape, n, n)
        Ls = [L.broadcast_to(*bshape, n, n) for L in Ls]
        rho0 = rho0.broadcast_to(*bshape, *rho0.shape[-2:])
        # vectorize the function
        f = multi_vmap(_mesolve_lr, in_axes, out_axes, nvmap)

    return f(
        H,
        Ls,
        rho0,
        tsave,
        exp_ops,
        method,
        gradient,
        options,
        M,
        normalize_each_eval,
        gram_reg,
        eps_init,
        key,
    )


def _mesolve_lr(
    H: TimeQArray,
    Ls: list[TimeQArray],
    rho0: QArray,
    tsave: Array,
    exp_ops: list[QArray] | None,
    method: Method,
    gradient: Gradient | None,
    options: Options,
    M: int,
    normalize_each_eval: bool,
    gram_reg: float,
    eps_init: float | None,
    key: PRNGKeyArray | None,
) -> MESolveLRResult:
    integrator_constructors = {
        Euler: mesolve_lr_euler_integrator_constructor,
        Dopri5: mesolve_lr_dopri5_integrator_constructor,
        Dopri8: mesolve_lr_dopri8_integrator_constructor,
        Tsit5: mesolve_lr_tsit5_integrator_constructor,
        Kvaerno3: mesolve_lr_kvaerno3_integrator_constructor,
        Kvaerno5: mesolve_lr_kvaerno5_integrator_constructor,
    }
    assert_method_supported(method, integrator_constructors.keys())
    integrator_constructor = integrator_constructors[type(method)]

    method.assert_supports_gradient(gradient)

    if rho0.isket():
        psi0 = rho0.to_jax()
        eps = 1e-4 if eps_init is None else eps_init
        m0 = initialize_m0_from_ket(psi0, M, eps=eps, key=key)
    else:
        rho0_dm = rho0.todm()
        eps = 1e-5 if eps_init is None else eps_init
        m0 = initialize_m0_from_dm(rho0_dm.to_jax(), M, eps=eps, key=key)

    Es = [E.to_jax() for E in exp_ops] if exp_ops is not None else None

    integrator = integrator_constructor(
        ts=tsave,
        y0=m0,
        method=method,
        gradient=gradient,
        result_class=MESolveLRResult,
        options=options,
        H=H,
        Ls=Ls,
        Es=Es,
        normalize_each_eval=normalize_each_eval,
        gram_reg=gram_reg,
        dims=rho0.dims,
    )

    return integrator.run()


def _check_mesolve_lr_args(
    H: TimeQArray,
    Ls: list[TimeQArray],
    rho0: QArray,
    exp_ops: list[QArray] | None,
    M: int,
    gram_reg: float,
    eps_init: float | None,
):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check Ls shape
    for i, L in enumerate(Ls):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})

    # === check rho0 shape and layout
    check_shape(rho0, 'rho0', '(..., n, 1)', '(..., n, n)', subs={'...': '...rho0'})
    check_qarray_is_dense(rho0, 'rho0')

    # === check exp_ops shape
    if exp_ops is not None:
        for i, E in enumerate(exp_ops):
            check_shape(E, f'exp_ops[{i}]', '(n, n)')

    if not isinstance(M, int):
        raise TypeError('Argument `M` must be an int.')
    if M <= 0:
        raise ValueError('Argument `M` must be a positive integer.')

    n = rho0.shape[-2]
    if M > n:
        raise ValueError(
            f'Argument `M` must be <= n (got M={M} and n={n}).'
        )

    if gram_reg < 0.0:
        raise ValueError('Argument `gram_reg` must be non-negative.')
    if eps_init is not None and eps_init < 0.0:
        raise ValueError('Argument `eps_init` must be non-negative.')
