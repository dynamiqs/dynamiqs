from __future__ import annotations

import warnings

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
    linear_solver: str = 'lineax',
    save_factors_only: bool = False,
    save_low_rank_chi: bool = False,
    eps_init: float | None = None,
    key: PRNGKeyArray | None = None,
) -> MESolveLRResult:
    r"""Solve the low rank Lindblad master equation.

    This function computes the evolution of the density matrix
    $\rho(t)=m(t)m(t)^\dagger$ at time $t$, starting from an initial state $\rho_0$,
    according to the low rank Lindblad master equation (with $\hbar=1$ and where time is
    implicit(1))
    $$
        \frac{\dd m}{\dt} = -iHm
        + \frac{1}{2}\sum_{k=1}^N \left(
            L_k m(m^{-1}L_k m)^\dag
            - L_k^\dag L_k m
        \right),
    $$
    where $H$ is the system's Hamiltonian and $\{L_k\}$ is a collection of jump
    operators. Reference: Goutte, Savona (2025) arxiv:2508.18114
    { .annotate }

    1. With explicit time dependence:
        - $\rho\to\rho(t)$
        - $H\to H(t)$
        - $L_k\to L_k(t)$

    Args:
        H (qarray-like or timeqarray of shape (...H, n, n)): Hamiltonian.
        jump_ops (list of qarray-like or timeqarray, each of shape (...Lk, n, n)):
            List of jump operators.
        rho0 (qarray-like of shape (...rho0, n, 1) or (...rho0, n, n)): Initial state.
        tsave (array-like of shape (ntsave,)): Times at which the states and
            expectation values are saved. The equation is solved from `tsave[0]` to
            `tsave[-1]`, or from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        M (int): Rank of the low-rank approximation, number of columns of m(t).
        exp_ops (list of qarray-like, each of shape (n, n), optional): List of
            operators for which the expectation value is computed.
        method: Method for the integration. Defaults to
            [`dq.method.Tsit5`][dynamiqs.method.Tsit5] (supported:
            [`Tsit5`][dynamiqs.method.Tsit5], [`Dopri5`][dynamiqs.method.Dopri5],
            [`Dopri8`][dynamiqs.method.Dopri8],
            [`Kvaerno3`][dynamiqs.method.Kvaerno3],
            [`Kvaerno5`][dynamiqs.method.Kvaerno5],
            [`Euler`][dynamiqs.method.Euler],
            [`Expm`][dynamiqs.method.Expm].
        gradient: Algorithm used to compute the gradient. The default is
            method-dependent, refer to the documentation of the chosen method for more
            details.
        options: Generic options (supported: `save_states`, `cartesian_batching`,
            `progress_meter`, `t0`, `save_extra`).
            ??? "Detailed options API"

                ```
                dq.Options(
                    save_states: bool = True,
                    cartesian_batching: bool = True,
                    progress_meter: AbstractProgressMeter | bool | None = None,
                    t0: ScalarLike | None = None,
                    save_extra: Callable[[Array], PyTree] | None = None,
                )
                ```

                **Parameters:**

                - **`save_states`** - If `True`, the state is saved at every time in
                    `tsave`, otherwise only the final state is returned.
                - **`cartesian_batching`** - If `True`, batched arguments are treated as
                    separated batch dimensions, otherwise the batching is performed over
                    a single shared batched dimension.
                - **`progress_meter`** - Progress meter indicating how far the solve has
                    progressed. Defaults to `None` which uses the global default
                    progress meter (see
                    [`dq.set_progress_meter()`][dynamiqs.set_progress_meter]). Set to
                    `True` for a [tqdm](https://github.com/tqdm/tqdm) progress meter,
                    and `False` for no output. See other options in
                    [dynamiqs/progress_meter.py](https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/progress_meter.py).
                    If gradients are computed, the progress meter only displays during
                    the forward pass.
                - **`t0`** - Initial time. If `None`, defaults to the first time in
                    `tsave`.
                - **`save_extra`** _(function, optional)_ - A function with signature
                    `f(QArray) -> PyTree` that takes a state as input and returns a
                    PyTree. This can be used to save additional arbitrary data
                    during the integration, accessible in `result.extra`.
        normalize_each_eval (bool): If `True`, the low-rank factors `m(t)` are
            normalized at each evaluation of the vector field. This may improve
            numerical stability. Defaults to `True`.
        linear_solver (str): Linear solver to use to solve the linear systems for
            computing the low-rank evolution. Supported values are:
            - `'lineax'`: Use the Lineax QR-based solver.
            - `'cholesky'`: Use a Cholesky-based solver. This can be more efficient
              for small ranks, but may fail if precision is set to single. Defaults to
              `'lineax'`.
        save_factors_only (bool): If `True`, only the low-rank factors `m(t)` are
            saved instead of the full density matrix `rho(t)`. This saves memory when
            `M << n`. Defaults to `False`.
        save_low_rank_chi (bool): If `True`, the low-rank accuracy metric $\chi(t)$ is
            saved at each time in `tsave`. Defaults to `False`.
        eps_init (float, optional): Regularization parameter for the initialization
            of the low-rank factors `m0`. When initializing from a pure state
            `psi0`, the factors are initialized as `m0 = [psi0, v1, ..., v(M-1)]` where
            `v1, ..., v(M-1)` are random orthonormal vectors scaled by `eps_init`.
            When initializing from a mixed state `rho0`, the factors are initialized
            from the leading `M` eigenvectors of `rho0 + eps_init * I`. If `None`,
            a default value is used (`1e-4` for pure states, `1e-5` for mixed states).
        key (PRNGKeyArray, optional): JAX PRNG key used for random initialization
            of the low-rank factors `m0`. If `None`, a default key is used.

    Returns:
        `dq.MESolveLRResult` object holding the result of the low rank
            Lindblad master equation integration. Use `result.states` to access the
            saved states and `result.expects` to access the saved expectation values.

            ??? "Detailed result API"
                ```python
                dq.MESolveLRResult
                ```

                **Attributes:**

                - **`states`** _(qarray of shape (..., nsave, n, n))_ - Saved states
                    with `nsave = ntsave`, or `nsave = 1` if
                    `options.save_states=False`.
                - **`final_state`** _(qarray of shape (..., n, n))_ - Saved final state.
                - **`expects`** _(array of shape (..., len(exp_ops), ntsave) or None)_ -
                    Saved expectation values, if specified by `exp_ops`.
                - **`extra`** _(PyTree or None)_ - Extra data saved with `save_extra()`
                    if specified in `options`.
                - **`infos`** _(PyTree or None)_ - Method-dependent information on the
                    resolution.
                - **`tsave`** _(array of shape (ntsave,))_ - Times for which results
                    were saved.
                - **`method`** _(Method)_ - Method used.
                - **`gradient`** _(Gradient)_ - Gradient used.
                - **`options`** _(Options)_ - Options used.
                - **`factors`** _(array of shape (..., nsave, n, M))_ - Saved low-rank
                    factors `m(t)`, if `save_factors_only=True`.
                - **`low_rank_chi`** _(array of shape (..., ntsave))_ - Saved $\chi(t)$,
                    if save_low_rank_chi=True`.

    Examples:
        ```python
           import dynamiqs as dq
           import jax.numpy as jnp

            n = 16
            M = 8
            a = dq.destroy(n)

            H = a.dag() @ a
            jump_ops = [a]
            psi0 = dq.coherent(n, 1.0)
            tsave = jnp.linspace(0, 1.0, 11)

            result = dq.mesolve_lr(H, jump_ops, psi0, tsave, M=M)
            print(result)
        ```

        ```text title="Output"
        |██████████| 100.0% ◆ elapsed 4.05ms ◆ remaining 0.00ms
        ==== MESolveLRResult ====
        Method :    Tsit5
        Infos  :    59 steps (58 accepted, 1 rejected)
        States :    QArray complex128 (11, 16, 16) | 44.0 Kb
        ```

    # Advanced use-cases

    ## Defining a time-dependent Hamiltonian or jump operator

    If the Hamiltonian or the jump operators depend on time, they can be converted to
    timeqarrays using [`dq.pwc()`][dynamiqs.pwc],
    [`dq.modulated()`][dynamiqs.modulated], or
    [`dq.timecallable()`][dynamiqs.timecallable]. See the
    [Time-dependent operators](../../documentation/basics/time-dependent-operators.md)
    tutorial for more details.

    ## Running multiple simulations concurrently

    The Hamiltonian `H`, the jump operators `jump_ops` and the initial density matrix
    `rho0` can be batched to solve multiple master equations concurrently. All other
    arguments are common to every batch. The resulting states and expectation values
    are batched according to the leading dimensions of `H`, `jump_ops` and  `rho0`. The
    behaviour depends on the value of the `cartesian_batching` option.

    === "If `cartesian_batching = True` (default value)"
        The results leading dimensions are
        ```
        ... = ...H, ...L0, ...L1, (...), ...rho0
        ```

        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(4, 5, n, n), (6, n, n)]_,
        - `rho0` has shape _(7, n, n)_,

        then `result.states` has shape _(2, 3, 4, 5, 6, 7, ntsave, n, n)_.
    === "If `cartesian_batching = False`"
        The results leading dimensions are
        ```
        ... = ...H = ...L0 = ...L1 = (...) = ...rho0  # (once broadcasted)
        ```

        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(3, n, n), (2, 1, n, n)]_,
        - `rho0` has shape _(3, n, n)_,

        then `result.states` has shape _(2, 3, ntsave, n, n)_.

    See the
    [Batching simulations](../../documentation/basics/batching-simulations.md)
    tutorial for more details.
    """
    # === convert arguments
    H = astimeqarray(H)
    Ls = [astimeqarray(L) for L in jump_ops]
    rho0 = asqarray(rho0)
    try:
        M = int(M)
    except (TypeError, ValueError) as exc:
        raise TypeError('Argument `M` must be an int.') from exc
    if not isinstance(linear_solver, str):
        raise TypeError('Argument `linear_solver` must be a string.')
    linear_solver = linear_solver.lower()
    save_factors_only = bool(save_factors_only)
    save_low_rank_chi = bool(save_low_rank_chi)
    if eps_init is not None:
        eps_init = float(eps_init)
    tsave = jnp.asarray(tsave)
    if exp_ops is not None:
        exp_ops = [asqarray(E) for E in exp_ops] if len(exp_ops) > 0 else None

    # === check arguments
    _check_mesolve_lr_args(H, Ls, rho0, exp_ops, M, linear_solver, eps_init)
    tsave = check_times(tsave, 'tsave')
    check_options(options, 'mesolve_lr')
    options = options.initialise()

    if linear_solver == 'cholesky' and not jax.config.read('jax_enable_x64'):
        warnings.warn(
            'Using the Cholesky linear solver with single-precision dtypes can be '
            'numerically unstable; consider enabling double precision with '
            "`dq.set_precision('double')`.",
            stacklevel=2,
        )

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
            'linear_solver',
            'save_factors_only',
            'save_low_rank_chi',
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
        linear_solver,
        save_factors_only,
        save_low_rank_chi,
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
    linear_solver: str,
    save_factors_only: bool,
    save_low_rank_chi: bool,
    eps_init: float | None,
    key: PRNGKeyArray | None,
) -> MESolveLRResult:
    # vectorize input over H, Ls and rho0
    in_axes = (H.in_axes, [L.in_axes for L in Ls], 0, *(None,) * 12)
    out_axes = MESolveLRResult.out_axes()

    if options.cartesian_batching:
        nvmap = (H.ndim - 2, [L.ndim - 2 for L in Ls], rho0.ndim - 2, *(0,) * 12)
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
        linear_solver,
        save_factors_only,
        save_low_rank_chi,
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
    linear_solver: str,
    save_factors_only: bool,
    save_low_rank_chi: bool,
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
        linear_solver=linear_solver,
        save_factors_only=save_factors_only,
        save_low_rank_chi=save_low_rank_chi,
        dims=rho0.dims,
    )

    return integrator.run()


def _check_mesolve_lr_args(
    H: TimeQArray,
    Ls: list[TimeQArray],
    rho0: QArray,
    exp_ops: list[QArray] | None,
    M: int,
    linear_solver: str,
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
    if n < M:
        raise ValueError(f'Argument `M` must be <= n (got M={M} and n={n}).')

    if linear_solver not in ('lineax', 'cholesky'):
        raise ValueError(
            "Argument `linear_solver` must be 'lineax' or 'cholesky', "
            f'got {linear_solver!r}.'
        )
    if eps_init is not None and eps_init < 0.0:
        raise ValueError('Argument `eps_init` must be non-negative.')
