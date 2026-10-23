from __future__ import annotations

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp

from ..._checks import check_qarray_is_dense, check_shape
from ...gradient import Gradient, Implicit
from ...method import (
    DiffraxSteadyState,
    Dopri5,
    Dopri8,
    Euler,
    Kvaerno3,
    Kvaerno5,
    Method,
    Rouchon1,
    Rouchon2,
    Rouchon3,
    Tsit5,
)
from ...options import Options, check_options
from ...qarrays.layout import dense
from ...qarrays.qarray import QArray, QArrayLike
from ...qarrays.utils import asqarray
from ...result import MESteadyStateResult
from ...time_qarray import TimeQArray
from ...utils.operators import eye
from .._utils import (
    assert_method_supported,
    astimeqarray,
    broadcast_batched,
    cartesian_vmap,
    catch_xla_runtime_error,
    multi_vmap,
)
from ..core.diffrax_integrator import (
    mesolve_dopri5_integrator_constructor,
    mesolve_dopri8_integrator_constructor,
    mesolve_euler_integrator_constructor,
    mesolve_kvaerno3_integrator_constructor,
    mesolve_kvaerno5_integrator_constructor,
    mesolve_tsit5_integrator_constructor,
)
from ..core.rouchon_integrator import (
    mesolve_rouchon1_integrator_constructor,
    mesolve_rouchon2_integrator_constructor,
    mesolve_rouchon3_integrator_constructor,
)
from ..core.steadystate_integrator import MESteadyStateIntegrator


def mesteadystate(
    H: QArrayLike | TimeQArray,
    jump_ops: list[QArrayLike | TimeQArray],
    *,
    rho0: QArrayLike | None = None,
    exp_ops: list[QArrayLike] | None = None,
    method: Method = DiffraxSteadyState(),  # noqa: B008
    gradient: Gradient | None = None,
    cartesian_batching: bool = True,
    vectorized: bool = False,
    assume_hermitian: bool = True,
) -> MESteadyStateResult:
    r"""Compute the steady state of the Lindblad master equation.

    This function computes the steady state $\rho_\infty$ of the Lindblad master
    equation (with $\hbar=1$), defined by
    $$
        \mathcal{L}(\rho_\infty) = -i[H, \rho_\infty]
        + \sum_{k=1}^N \left(
            L_k \rho_\infty L_k^\dag
            - \frac{1}{2} L_k^\dag L_k \rho_\infty
            - \frac{1}{2} \rho_\infty L_k^\dag L_k
        \right) = 0,
    $$
    where $H$ is the system's Hamiltonian and $\{L_k\}$ is a collection of jump
    operators. If the steady state is not unique, the one returned depends on the
    initial state `rho0` of the method.

    Warning:
        Time-dependent Hamiltonian and jump operators are accepted but not checked
        for. A time-dependent Lindbladian has no steady state in general: the method
        then stops at the first time where $\mathcal{L}(t)(\rho)$ is small, and the
        default gradient [`dq.gradient.Implicit`][dynamiqs.gradient.Implicit] is
        wrong.

    Args:
        H (qarray-like or timeqarray of shape (...H, n, n)): Hamiltonian.
        jump_ops (list of qarray-like or timeqarray, each of shape (...Lk, n, n)):
            List of jump operators.
        rho0 (qarray-like of shape (...rho0, n, 1) or (...rho0, n, n), optional):
            Initial state of the method. Defaults to the maximally mixed state
            $I/n$, which converges to the mixture of all steady states when the
            steady state is not unique.
        exp_ops (list of qarray-like, each of shape (n, n), optional): List of
            operators for which the expectation value is computed.
        method: Method to compute the steady state. Defaults to
            [`dq.method.DiffraxSteadyState`][dynamiqs.method.DiffraxSteadyState]
            (supported: [`DiffraxSteadyState`][dynamiqs.method.DiffraxSteadyState]).
        gradient: Algorithm used to compute the gradient. Defaults to
            [`dq.gradient.Implicit`][dynamiqs.gradient.Implicit], refer to the
            documentation of the chosen method for the other supported gradients.

    Returns:
        `dq.MESteadyStateResult` object holding the steady state. Use `result.state`
            to access the steady state and `result.expects` to access its expectation
            values.

            ??? "Detailed result API"
                ```python
                dq.MESteadyStateResult
                ```

                **Attributes:**

                - **`state`** _(qarray of shape (..., n, n))_ - Steady state, or last
                    state of the method if it did not converge.
                - **`expects`** _(array of shape (..., len(exp_ops)) or None)_ -
                    Expectation values of the steady state, if specified by
                    `exp_ops`.
                - **`residual`** _(array of shape (...))_ - Root mean square of the
                    matrix elements of $\mathcal{L}(\rho)$ for the returned state.
                - **`infos`** _(PyTree)_ - Information on the resolution:
                    `infos.converged` _(array of shape (...))_ is `True` if the
                    steady state was reached, `infos.time` _(array of shape (...))_
                    is the time at which the method stopped, and `infos.steps` holds
                    the number of steps.
                - **`method`** _(Method)_ - Method used.
                - **`gradient`** _(Gradient)_ - Gradient used.
                - **`options`** _(Options)_ - Options used.

    Other Parameters:
        cartesian_batching: If `True`, batched arguments are treated
            as separated batch dimensions, otherwise the batching is performed over a
            single shared batch dimension. Defaults to `True`.
        vectorized: If `True`, the master equation is solved by
            vectorizing the density matrix and Liouvillian, see
            [`dq.mesolve()`][dynamiqs.mesolve]. Defaults to `False`.
        assume_hermitian: If `True`, the initial density matrix
            `rho0` is assumed to be Hermitian, see
            [`dq.mesolve()`][dynamiqs.mesolve]. Defaults to `True`.

    Examples:
        ```python
        import dynamiqs as dq

        n = 16
        a = dq.destroy(n)

        H = a.dag() @ a + 0.5 * (a + a.dag())
        jump_ops = [a]

        result = dq.mesteadystate(H, jump_ops, exp_ops=[a.dag() @ a])
        print(result)
        ```

        ```text title="Output"
        ==== MESteadyStateResult ====
        Method   : DiffraxSteadyState
        Infos    : converged at t=18.5 | 32 steps (32 accepted, 0 rejected)
        State    : QArray complex64 (16, 16) | 2.0 Kb
        Expects  : Array complex64 (1,) | 0.0 Kb
        Residual : Array float32 () | 0.0 Kb
        ```

    # Advanced use-cases

    ## Running multiple steady state computations concurrently

    The Hamiltonian `H`, the jump operators `jump_ops` and the initial density matrix
    `rho0` can be batched to compute multiple steady states concurrently, as in
    [`dq.mesolve()`][dynamiqs.mesolve]. The resulting states and expectation values
    are batched according to the leading dimensions of `H`, `jump_ops` and `rho0`.

    === "If `cartesian_batching = True` (default value)"
        The results leading dimensions are
        ```
        ... = ...H, ...L0, ...L1, (...), ...rho0
        ```

        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(4, 5, n, n), (6, n, n)]_,
        - `rho0` has shape _(7, n, n)_,

        then `result.state` has shape _(2, 3, 4, 5, 6, 7, n, n)_.
    === "If `cartesian_batching = False`"
        The results leading dimensions are
        ```
        ... = ...H = ...L0 = ...L1 = (...) = ...rho0  # (once broadcasted)
        ```

        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(3, n, n), (2, 1, n, n)]_,
        - `rho0` has shape _(3, n, n)_,

        then `result.state` has shape _(2, 3, n, n)_.

    See the
    [Batching simulations](../../documentation/basics/batching-simulations.md)
    tutorial for more details.
    """
    # === convert arguments
    H = astimeqarray(H, 'H')
    Ls = [astimeqarray(L, 'jump_ops') for L in jump_ops]
    if rho0 is None:
        n = H.shape[-1]
        rho0 = eye(*H.dims, layout=dense) / n
    rho0 = asqarray(rho0)

    _exp_ops = None
    if exp_ops is not None:
        _exp_ops = [asqarray(E) for E in exp_ops] if len(exp_ops) > 0 else None

    # === build options
    options = Options(
        cartesian_batching=cartesian_batching,
        vectorized=vectorized,
        assume_hermitian=assume_hermitian,
    )

    # === check arguments
    _check_mesteadystate_args(H, Ls, rho0, _exp_ops)
    check_options(options, 'mesteadystate')
    # no progress meter: it shows the fraction of [t0, t1], with t1 = inf
    options = replace(options, progress_meter=False).initialise()

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to qarrays
    f = jax.jit(_vectorized_mesteadystate, static_argnames=('gradient', 'options'))
    return f(H, Ls, rho0, _exp_ops, method, gradient, options)


@catch_xla_runtime_error
def _vectorized_mesteadystate(
    H: TimeQArray,
    Ls: list[TimeQArray],
    rho0: QArray,
    exp_ops: list[QArray] | None,
    method: Method,
    gradient: Gradient | None,
    options: Options,
) -> MESteadyStateResult:
    # vectorize input over H, Ls and rho0
    in_axes = (H.in_axes, [L.in_axes for L in Ls], 0, None, None, None, None)
    out_axes = MESteadyStateResult.out_axes()

    if options.cartesian_batching:
        nvmap = (H.ndim - 2, [L.ndim - 2 for L in Ls], rho0.ndim - 2, 0, 0, 0, 0)
        f = cartesian_vmap(_mesteadystate, in_axes, out_axes, nvmap)
    else:
        bshape = jnp.broadcast_shapes(*[x.shape[:-2] for x in [H, *Ls, rho0]])
        nvmap = len(bshape)
        # broadcast batched inputs to the same shape, keep unbatched inputs unbatched
        H, H_axes = broadcast_batched(H, H.in_axes, bshape)
        Ls_and_axes = [broadcast_batched(L, L.in_axes, bshape) for L in Ls]
        Ls = [L for L, _ in Ls_and_axes]
        rho0, rho0_axes = broadcast_batched(rho0, 0, bshape)
        in_axes = (H_axes, [axes for _, axes in Ls_and_axes], rho0_axes, *in_axes[3:])
        # vectorize the function
        f = multi_vmap(_mesteadystate, in_axes, out_axes, nvmap)

    return f(H, Ls, rho0, exp_ops, method, gradient, options)


def _mesteadystate(
    H: TimeQArray,
    Ls: list[TimeQArray],
    rho0: QArray,
    exp_ops: list[QArray] | None,
    method: Method,
    gradient: Gradient | None,
    options: Options,
) -> MESteadyStateResult:
    # === select integrator constructor
    assert_method_supported(method, [DiffraxSteadyState])
    integrator_constructors = {
        Euler: mesolve_euler_integrator_constructor,
        Rouchon1: mesolve_rouchon1_integrator_constructor,
        Rouchon2: mesolve_rouchon2_integrator_constructor,
        Rouchon3: mesolve_rouchon3_integrator_constructor,
        Dopri5: mesolve_dopri5_integrator_constructor,
        Dopri8: mesolve_dopri8_integrator_constructor,
        Tsit5: mesolve_tsit5_integrator_constructor,
        Kvaerno3: mesolve_kvaerno3_integrator_constructor,
        Kvaerno5: mesolve_kvaerno5_integrator_constructor,
    }
    me_method = cast(DiffraxSteadyState, method).me_method
    integrator_constructor = integrator_constructors[type(me_method)]  # ty: ignore

    # === check gradient is supported
    implicit = isinstance(gradient, Implicit | None)
    if not implicit:
        me_method.assert_supports_gradient(gradient)

    # === init integrator
    integrator = integrator_constructor(
        ts=jnp.array([0.0, jnp.inf]),
        y0=rho0,
        method=me_method,
        gradient=None if implicit else gradient,
        options=options,
        H=H,
        Ls=Ls,
        Es=None,
    )
    steadystate_integrator = MESteadyStateIntegrator(
        integrator=integrator,
        method=cast(DiffraxSteadyState, method),
        gradient=gradient,
        options=options,
        Es=exp_ops,
    )

    # === run integrator
    return steadystate_integrator.run()


def _check_mesteadystate_args(
    H: TimeQArray, Ls: list[TimeQArray], rho0: QArray, exp_ops: list[QArray] | None
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
