from __future__ import annotations

from collections.abc import Callable
from typing import cast

import equinox as eqx
from jaxtyping import PyTree

from ..._checks import check_qarray_is_dense, check_shape
from ...gradient import Gradient
from ...method import Method, TimeEvolution
from ...options import Options, check_options
from ...qarrays.layout import dense
from ...qarrays.qarray import QArray, QArrayLike
from ...qarrays.utils import asqarray
from ...result import SteadyStateResult
from ...time_qarray import ConstantTimeQArray, TimeQArray, _unclipped
from ...utils.operators import eye
from .._utils import assert_method_supported, astimeqarray

__all__ = ['steadystate']


def steadystate(
    H: QArrayLike | ConstantTimeQArray,
    jump_ops: list[QArrayLike | ConstantTimeQArray],
    *,
    exp_ops: list[QArrayLike] | None = None,
    method: Method = TimeEvolution(),  # noqa: B008
    gradient: Gradient | None = None,
    cartesian_batching: bool = True,
    save_extra: Callable[[QArray], PyTree] | None = None,
    vectorized: bool = False,
    assume_hermitian: bool = True,
) -> SteadyStateResult:
    r"""Compute the steady state of the Lindblad master equation.

    This function computes a steady state $\rho_\infty$ of the Lindblad master
    equation (with $\hbar=1$), defined by $\mathcal{L}(\rho_\infty) = 0$ with
    $$
        \mathcal{L}(\rho) = -i[H, \rho]
        + \sum_{k=1}^N \left(
            L_k \rho L_k^\dag
            - \frac{1}{2} L_k^\dag L_k \rho
            - \frac{1}{2} \rho L_k^\dag L_k
        \right),
    $$
    where $H$ is the system's Hamiltonian and $\{L_k\}$ is a collection of jump
    operators, both time-independent.

    Args:
        H (qarray-like or constant timeqarray of shape (...H, n, n)): Hamiltonian.
        jump_ops (list of qarray-like or constant timeqarray, each of shape
            (...Lk, n, n)): List of jump operators, must not be empty.
        exp_ops (list of qarray-like, each of shape (n, n), optional): List of
            operators for which the expectation value is computed.
        method: Method for the computation. Defaults to
            [`dq.method.TimeEvolution`][dynamiqs.method.TimeEvolution] (supported:
            [`TimeEvolution`][dynamiqs.method.TimeEvolution]).
        gradient: Algorithm used to compute the gradient. The default is
            method-dependent, refer to the documentation of the chosen method for more
            details.

    Returns:
        `dq.SteadyStateResult` object holding the result of the steady state
            computation. Use `result.state` to access the steady state,
            `result.expects` to access its expectation values, and
            `result.infos.converged` to check that the method converged.

            ??? "Detailed result API"
                ```python
                dq.SteadyStateResult
                ```

                **Attributes:**

                - **`state`** _(qarray of shape (..., n, n))_ - Steady state, or last
                    state reached if the method did not converge.
                - **`expects`** _(array of shape (..., len(exp_ops)) or None)_ -
                    Expectation values in the steady state, if specified by
                    `exp_ops`.
                - **`extra`** _(PyTree or None)_ - Extra data saved with
                    `save_extra()` if specified.
                - **`infos`** _(PyTree)_ - Information on the resolution:
                    `infos.converged` _(bool array of shape (...))_ is `True` if the
                    convergence criterion of the method was met, `infos.time`
                    _(array of shape (...))_ is the time at which the method stopped,
                    and `infos.steps` holds the number of steps.
                - **`method`** _(Method)_ - Method used.
                - **`gradient`** _(Gradient)_ - Gradient used.
                - **`options`** _(Options)_ - Options used.

    Other Parameters:
        cartesian_batching: If `True`, batched arguments are treated as separated
            batch dimensions, otherwise the batching is performed over a single shared
            batch dimension. Defaults to `True`.
        save_extra: Function with signature `f(QArray) -> PyTree` that takes the
            steady state as input and returns a PyTree, accessible in
            `result.extra`. Defaults to `None`.
        vectorized: If `True`, the master equation is solved by vectorizing the
            density matrix and Liouvillian, see [`dq.mesolve()`][dynamiqs.mesolve].
            Defaults to `False`.
        assume_hermitian: If `True`, the initial density matrix of the method is
            assumed to be Hermitian, see [`dq.mesolve()`][dynamiqs.mesolve].
            Defaults to `True`.

    # Advanced use-cases

    ## Running multiple computations concurrently

    The Hamiltonian `H` and the jump operators `jump_ops` can be batched to compute
    multiple steady states concurrently. All other arguments are common to every
    batch. The resulting states and expectation values are batched according to the
    leading dimensions of `H` and `jump_ops`. The behaviour depends on the value of
    the `cartesian_batching` option.

    === "If `cartesian_batching = True` (default value)"
        The results leading dimensions are
        ```
        ... = ...H, ...L0, ...L1, (...)
        ```

        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(4, 5, n, n), (6, n, n)]_,

        then `result.state` has shape _(2, 3, 4, 5, 6, n, n)_.
    === "If `cartesian_batching = False`"
        The results leading dimensions are
        ```
        ... = ...H = ...L0 = ...L1 = (...)  # (once broadcast)
        ```

        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(3, n, n), (2, 1, n, n)]_,

        then `result.state` has shape _(2, 3, n, n)_.

    See the
    [Batching simulations](../../documentation/basics/batching-simulations.md)
    tutorial for more details.
    """
    # === convert arguments
    _H = astimeqarray(H, 'H')
    Ls = [astimeqarray(L, 'jump_ops') for L in jump_ops]

    _exp_ops = None
    if exp_ops is not None:
        _exp_ops = [asqarray(E) for E in exp_ops] if len(exp_ops) > 0 else None

    # === build options (the progress over [0, ∞) is meaningless, so it is disabled)
    options = Options(
        cartesian_batching=cartesian_batching,
        progress_meter=False,
        save_extra=save_extra,
        vectorized=vectorized,
        assume_hermitian=assume_hermitian,
    )

    # === check arguments
    _check_steadystate_args(_H, Ls, _exp_ops)
    check_options(options, 'steadystate')
    options = options.initialise()
    method = _init_method(method, _H)

    # === check gradient is supported
    method.assert_supports_gradient(gradient)
    method.ode_method.assert_supports_gradient(gradient)

    # todo: evolve rho0 with `ode_method` until the stopping criterion of the method
    # is met
    raise NotImplementedError('`dq.steadystate()` is not implemented yet.')


def _init_method(method: Method, H: TimeQArray) -> TimeEvolution:
    assert_method_supported(method, [TimeEvolution])
    method = cast(TimeEvolution, method)

    # === default to the maximally mixed state, which overlaps every steady state
    if method.rho0 is None:
        rho0 = eye(*H.dims, layout=dense) / H.shape[-1]
        return eqx.tree_at(lambda m: m.rho0, method, rho0, is_leaf=lambda x: x is None)

    check_shape(method.rho0, 'method.rho0', '(n, 1)', '(n, n)')
    check_qarray_is_dense(method.rho0, 'method.rho0')
    return method


def _check_steadystate_args(
    H: TimeQArray, Ls: list[TimeQArray], exp_ops: list[QArray] | None
):
    # === check H and Ls are time-independent
    _check_time_independent(H, 'H')
    for i, L in enumerate(Ls):
        _check_time_independent(L, f'jump_ops[{i}]')

    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check Ls shape
    if len(Ls) == 0:
        raise ValueError(
            'Argument `jump_ops` must contain at least one jump operator, but is an'
            ' empty list. Without dissipation, the steady state is not unique.'
        )
    for i, L in enumerate(Ls):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})

    # === check exp_ops shape
    if exp_ops is not None:
        for i, E in enumerate(exp_ops):
            check_shape(E, f'exp_ops[{i}]', '(n, n)')


def _check_time_independent(x: TimeQArray, argname: str):
    # a clipped constant vanishes outside its time window
    if not (isinstance(x, ConstantTimeQArray) and _unclipped(x)):
        kind = 'clipped ' if isinstance(x, ConstantTimeQArray) else ''
        raise TypeError(
            f'Argument `{argname}` must be time-independent (a qarray-like or an'
            f' unclipped constant timeqarray), but is a {kind}`{type(x).__name__}`.'
        )
