from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from ..._checks import check_shape, check_times
from ...gradient import Gradient
from ...method import EulerJump, Event, Method
from ...options import Options, check_options
from ...qarrays.qarray import QArray, QArrayLike
from ...qarrays.utils import asqarray
from ...result import JSSESolveResult
from ...time_qarray import TimeQArray
from .._utils import (
    assert_method_supported,
    astimeqarray,
    cartesian_vmap,
    catch_xla_runtime_error,
    multi_vmap,
)
from ..core.event_integrator import jssesolve_event_integrator_constructor
from ..core.fixed_step_stochastic_integrator import (
    jssesolve_euler_jump_integrator_constructor,
)


def jssesolve(
    H: QArrayLike | TimeQArray,
    jump_ops: list[QArrayLike | TimeQArray],
    psi0: QArrayLike,
    tsave: ArrayLike,
    keys: PRNGKeyArray,
    *,
    exp_ops: list[QArrayLike] | None = None,
    method: Method | None = None,
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> JSSESolveResult:
    r"""Solve the jump stochastic Schrödinger equation (SSE).

    The jump SSE describes the evolution of a quantum system measured by an ideal jump
    detector (for example photodetection in quantum optics). This
    function computes the evolution of the state vector $\ket{\psi(t)}$ at time $t$,
    starting from an initial state $\ket{\psi_0}$, according to the jump SSE ($\hbar=1$,
    time is implicit(1))
    $$
        \dd\\!\ket\psi = \left[
            -iH \dt
            - \frac12 \sum_{k=1}^N \left(
                L_k^\dag L_k - \braket{L_k^\dag L_k}
            \right) \dt
            + \sum_{k=1}^N \left(
                \frac{L_k}{\sqrt{\braket{L_k^\dag L_k}}} - 1
            \right) \dd N_k
        \right] \\!\ket\psi
    $$
    where $H$ is the system's Hamiltonian, $\{L_k\}$ is a collection of jump operators,
    each continuously measured with perfect efficiency, and $\dd N_k$ are independent
    point processes with law
    $$
        \begin{split}
            \mathbb{P}[\dd N_k = 0] &= 1 - \mathbb{P}[\dd N_k = 1], \\\\
            \mathbb{P}[\dd N_k = 1] &= \braket{L_k^\dag L_k} \dt.
        \end{split}
    $$
    { .annotate }

    1. With explicit time dependence:
        - $\ket\psi\to\ket{\psi(t)}$
        - $H\to H(t)$
        - $L_k\to L_k(t)$
        - $\dd N_k\to \dd N_k(t)$

    The continuous-time measurements are defined by the point processes $\dd N_k$. The
    solver returns the times at which the detector clicked,
    $I_k = \{t \in [t_0, t_\text{end}[ \,|\, \dd N_k(t)=1\}$.

    Note:
        If you are only interested in simulating trajectories to solve the Lindblad
        master equation, consider using [`dq.mesolve()`][dynamiqs.mesolve] with the
        [`dq.method.JumpMonteCarlo`][dynamiqs.method.JumpMonteCarlo] method.

    Note:
        This function is the Dynamiqs counterpart of QuTiP's `mcsolve()` function.

    Args:
        H _(qarray-like or time-qarray of shape (...H, n, n))_: Hamiltonian.
        jump_ops _(list of qarray-like or time-qarray, each of shape (...Lk, n, n))_:
            List of jump operators.
        psi0 _(qarray-like of shape (...psi0, n, 1))_: Initial state.
        tsave _(array-like of shape (ntsave,))_: Times at which the states and
            expectation values are saved. The equation is solved from `tsave[0]` to
            `tsave[-1]`.
        keys _(list of PRNG keys)_: PRNG keys used to sample the point processes.
            The number of elements defines the number of sampled stochastic
            trajectories.
        exp_ops _(list of array-like, each of shape (n, n), optional)_: List of
            operators for which the expectation value is computed.
        method: Method for the integration. No defaults for now, you have to specify a
            method (supported: [`Event`][dynamiqs.method.Event],
            [`EulerJump`][dynamiqs.method.EulerJump]).
        gradient: Algorithm used to compute the gradient. The default is
            method-dependent, refer to the documentation of the chosen method for more
            details.
        options: Generic options (supported: `save_states`, `cartesian_batching`, `t0`,
            `save_extra`, `nmaxclick`).
            ??? "Detailed options API"
                ```
                dq.Options(
                    save_states: bool = True,
                    cartesian_batching: bool = True,
                    t0: ScalarLike | None = None,
                    save_extra: callable[[Array], PyTree] | None = None,
                    nmaxclick: int = 10_000,
                )
                ```

                **Parameters**

                - **save_states** - If `True`, the state is saved at every time in
                    `tsave`, otherwise only the final state is returned.
                - **cartesian_batching** - If `True`, batched arguments are treated as
                    separated batch dimensions, otherwise the batching is performed over
                    a single shared batched dimension.
                - **t0** - Initial time. If `None`, defaults to the first time in
                    `tsave`.
                - **save_extra** _(function, optional)_ - A function with signature
                    `f(QArray) -> PyTree` that takes a state as input and returns a
                    PyTree. This can be used to save additional arbitrary data
                    during the integration, accessible in `result.extra`.
                - **nmaxclick** - Maximum buffer size for `result.clicktimes`, should be
                    set higher than the expected maximum number of clicks.

    Returns:
        `dq.JSSESolveResult` object holding the result of the jump SSE integration. Use
            `result.states` to access the saved states, `result.expects` to access the
            saved expectation values and `result.clicktimes` to access the detector
            click times.

            ??? "Detailed result API"
                ```python
                dq.JSSESolveResult
                ```

                For the shape indications we define `ntrajs` as the number of
                trajectories (`ntrajs = len(keys)`).

                **Attributes**

                - **states** _(qarray of shape (..., ntrajs, nsave, n, 1))_ - Saved
                    states with `nsave = ntsave`, or `nsave = 1` if
                    `options.save_states=False`.
                - **final_state** _(qarray of shape (..., ntrajs, n, 1))_ - Saved final
                    state.
                - **expects** _(array of shape (..., ntrajs, len(exp_ops), ntsave) or None)_ - Saved
                    expectation values, if specified by `exp_ops`.
                - **clicktimes** _(array of shape (..., ntrajs, len(jump_ops), nmaxclick))_ - Times
                    at which the detectors clicked. Variable-length array padded with
                    `jnp.nan` up to `nmaxclick`.
                - **nclicks** _(array of shape (..., ntrajs, len(jump_ops))_ - Number
                    of clicks for each jump operator.
                - **extra** _(PyTree or None)_ - Extra data saved with `save_extra()` if
                    specified in `options`.
                - **keys** _(PRNG key array of shape (ntrajs,))_ - PRNG keys used to
                    sample the point processes.
                - **infos** _(PyTree or None)_ - Method-dependent information on the
                    resolution.
                - **tsave** _(array of shape (ntsave,))_ - Times for which results were
                    saved.
                - **method** _(Method)_ - Method used.
                - **gradient** _(Gradient)_ - Gradient used.
                - **options** _(Options)_ - Options used.

    Examples:
        ```python
        import dynamiqs as dq
        import jax.numpy as jnp
        import jax

        n = 16
        a = dq.destroy(n)

        H = a.dag() @ a
        jump_ops = [a]
        psi0 = dq.coherent(n, 1.0)
        tsave = jnp.linspace(0, 1.0, 11)
        keys = jax.random.split(jax.random.key(42), 100)

        method = dq.method.EulerJump(dt=1e-3)
        result = dq.jssesolve(H, jump_ops, psi0, tsave, keys, method=method)
        print(result)
        ```

        ```text title="Output"
        ==== JSSESolveResult ====
        Method     : EulerJump
        Infos      : 1000 steps | infos shape (100,)
        States     : QArray complex64 (100, 11, 16, 1) | 137.5 Kb
        Clicktimes : Array float32 (100, 1, 10000) | 3.8 Mb
        ```

    # Advanced use-cases

    ## Defining a time-dependent Hamiltonian or jump operator

    If the Hamiltonian or the jump operators depend on time, they can be converted
    to time-arrays using [`dq.pwc()`][dynamiqs.pwc],
    [`dq.modulated()`][dynamiqs.modulated], or
    [`dq.timecallable()`][dynamiqs.timecallable]. See the
    [Time-dependent operators](../../documentation/basics/time-dependent-operators.md)
    tutorial for more details.

    ## Running multiple simulations concurrently

    The Hamiltonian `H`, the jump operators `jump_ops` and the initial state `psi0` can
    be batched to solve multiple SSEs concurrently. All other arguments (including the
    PRNG key) are common to every batch. The resulting states, measurements and
    expectation values are batched according to the leading dimensions of `H`,
    `jump_ops` and `psi0`. The behaviour depends on the value of the
    `cartesian_batching` option.

    === "If `cartesian_batching = True` (default value)"
        The results leading dimensions are
        ```
        ... = ...H, ...L0, ...L1, (...), ...psi0
        ```
        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(4, 5, n, n), (6, n, n)]_,
        - `psi0` has shape _(7, n, 1)_,

        then `result.states` has shape _(2, 3, 4, 5, 6, 7, ntrajs, ntsave, n, 1)_.

    === "If `cartesian_batching = False`"
        The results leading dimensions are
        ```
        ... = ...H = ...L0 = ...L1 = (...) = ...psi0  # (once broadcasted)
        ```
        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(3, n, n), (2, 1, n, n)]_,
        - `psi0` has shape _(3, n, 1)_,

        then `result.states` has shape _(2, 3, ntrajs, ntsave, n, 1)_.

    See the
    [Batching simulations](../../documentation/basics/batching-simulations.md)
    tutorial for more details.
    """  # noqa: E501
    # === convert arguments
    H = astimeqarray(H)
    Ls = [astimeqarray(L) for L in jump_ops]
    psi0 = asqarray(psi0)
    tsave = jnp.asarray(tsave)
    keys = jnp.asarray(keys)
    if exp_ops is not None:
        exp_ops = [asqarray(E) for E in exp_ops] if len(exp_ops) > 0 else None

    # === check arguments
    _check_jssesolve_args(H, Ls, psi0, exp_ops)
    tsave = check_times(tsave, 'tsave')
    check_options(options, 'jssesolve')
    options = options.initialise()

    if method is None:
        raise ValueError('Argument `method` must be specified.')

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    f = _vectorized_jssesolve
    if isinstance(method, EulerJump):
        tsave = tuple(tsave.tolist())  # todo: fix static tsave
        f = jax.jit(f, static_argnames=('tsave', 'gradient', 'options'))
    else:
        f = jax.jit(f, static_argnames=('gradient', 'options'))
    return f(H, Ls, psi0, tsave, keys, exp_ops, method, gradient, options)


@catch_xla_runtime_error
def _vectorized_jssesolve(
    H: TimeQArray,
    Ls: list[TimeQArray],
    psi0: QArray,
    tsave: Array,
    keys: PRNGKeyArray,
    exp_ops: list[QArray] | None,
    method: Method,
    gradient: Gradient | None,
    options: Options,
) -> JSSESolveResult:
    # vectorize input over H, Ls and psi0
    in_axes = (H.in_axes, [L.in_axes for L in Ls], 0, *(None,) * 6)
    out_axes = JSSESolveResult.out_axes()

    if options.cartesian_batching:
        nvmap = (H.ndim - 2, [L.ndim - 2 for L in Ls], psi0.ndim - 2, 0, 0, 0, 0, 0, 0)
        f = cartesian_vmap(_jssesolve_many_trajectories, in_axes, out_axes, nvmap)
    else:
        bshape = jnp.broadcast_shapes(*[x.shape[:-2] for x in [H, *Ls, psi0]])
        nvmap = len(bshape)
        # broadcast all vectorized input to same shape
        n = H.shape[-1]
        H = H.broadcast_to(*bshape, n, n)
        Ls = [L.broadcast_to(*bshape, n, n) for L in Ls]
        psi0 = psi0.broadcast_to(*bshape, n, 1)
        # vectorize the function
        f = multi_vmap(_jssesolve_many_trajectories, in_axes, out_axes, nvmap)

    return f(H, Ls, psi0, tsave, keys, exp_ops, method, gradient, options)


def _jssesolve_many_trajectories(
    H: TimeQArray,
    Ls: list[TimeQArray],
    psi0: QArray,
    tsave: Array,
    keys: PRNGKeyArray,
    exp_ops: list[QArray] | None,
    method: Method,
    gradient: Gradient | None,
    options: Options,
) -> JSSESolveResult:
    f = _jssesolve_single_trajectory

    if isinstance(method, Event):
        # vectorization over keys is handled by the integrator
        pass
    else:
        # vectorize input over keys
        in_axes = (None, None, None, None, 0, None, None, None, None)
        out_axes = JSSESolveResult(None, None, None, None, 0, 0, 0)
        f = jax.vmap(f, in_axes, out_axes)

    return f(H, Ls, psi0, tsave, keys, exp_ops, method, gradient, options)


def _jssesolve_single_trajectory(
    H: TimeQArray,
    Ls: list[TimeQArray],
    psi0: QArray,
    tsave: Array,
    key: PRNGKeyArray,
    exp_ops: list[QArray] | None,
    method: Method,
    gradient: Gradient | None,
    options: Options,
) -> JSSESolveResult:
    # === select integrator constructor
    integrator_constructors = {
        EulerJump: jssesolve_euler_jump_integrator_constructor,
        Event: jssesolve_event_integrator_constructor,
    }
    assert_method_supported(method, integrator_constructors.keys())
    integrator_constructor = integrator_constructors[type(method)]

    # === check gradient is supported
    method.assert_supports_gradient(gradient)

    # === init integrator
    integrator = integrator_constructor(
        ts=tsave,
        y0=psi0,
        method=method,
        gradient=gradient,
        result_class=JSSESolveResult,
        options=options,
        H=H,
        Ls=Ls,
        Es=exp_ops,
        key=key,
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_jssesolve_args(
    H: TimeQArray, Ls: list[TimeQArray], psi0: QArray, exp_ops: list[QArray] | None
):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check Ls shape
    for i, L in enumerate(Ls):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})

    if len(Ls) == 0:
        raise ValueError(
            'Argument `jump_ops` is an empty list, consider using `dq.sesolve()` to'
            ' solve the Schrödinger equation.'
        )

    # === check psi0 shape
    check_shape(psi0, 'psi0', '(..., n, 1)', subs={'...': '...psi0'})

    # === check exp_ops shape
    if exp_ops is not None:
        for i, E in enumerate(exp_ops):
            check_shape(E, f'exp_ops[{i}]', '(n, n)')
