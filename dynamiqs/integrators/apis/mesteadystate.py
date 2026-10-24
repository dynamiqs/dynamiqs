from __future__ import annotations

from typing import cast

from ..._checks import check_qarray_is_dense, check_shape
from ...gradient import Gradient, Implicit
from ...method import DiffraxSteadyState, Method
from ...options import Options, check_options
from ...qarrays.layout import dense
from ...qarrays.qarray import QArray, QArrayLike
from ...qarrays.utils import asqarray
from ...result import MESteadyStateResult
from ...time_qarray import ConstantTimeQArray, TimeQArray, _unclipped
from ...utils.operators import eye
from .._utils import assert_method_supported, astimeqarray


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

    Args:
        H (qarray-like or constant timeqarray of shape (...H, n, n)): Hamiltonian.
        jump_ops (list of qarray-like or constant timeqarray of shape (...Lk, n, n)):
            List of jump operators.
        rho0 (qarray-like of shape (...rho0, n, 1) or (...rho0, n, n), optional):
            Initial state of the method. Defaults to the maximally mixed state
            $I/n$, which converges to a mixture of the steady states, weighted by the
            conserved quantities of the Lindbladian, when the steady state is not
            unique.
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
    options = options.initialise()

    return _mesteadystate(H, Ls, rho0, _exp_ops, method, gradient, options)


def _mesteadystate(
    H: TimeQArray,
    Ls: list[TimeQArray],
    rho0: QArray,
    exp_ops: list[QArray] | None,
    method: Method,
    gradient: Gradient | None,
    options: Options,
) -> MESteadyStateResult:
    # === check method is supported
    assert_method_supported(method, [DiffraxSteadyState])
    me_method = cast(DiffraxSteadyState, method).me_method

    # === check gradient is supported
    method.assert_supports_gradient(gradient)
    if not isinstance(gradient, Implicit):
        me_method.assert_supports_gradient(gradient)

    # todo: evolve rho0 with `me_method` until the stopping criterion of the method
    # is met, and differentiate the steady state implicitly by default
    raise NotImplementedError('`dq.mesteadystate()` is not implemented yet.')


def _check_mesteadystate_args(
    H: TimeQArray, Ls: list[TimeQArray], rho0: QArray, exp_ops: list[QArray] | None
):
    # === check H shape and time dependence
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})
    _check_time_independent(H, 'H')

    # === check Ls shape and time dependence
    for i, L in enumerate(Ls):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})
        _check_time_independent(L, f'jump_ops[{i}]')

    # === check rho0 shape and layout
    check_shape(rho0, 'rho0', '(..., n, 1)', '(..., n, n)', subs={'...': '...rho0'})
    check_qarray_is_dense(rho0, 'rho0')

    # === check exp_ops shape
    if exp_ops is not None:
        for i, E in enumerate(exp_ops):
            check_shape(E, f'exp_ops[{i}]', '(n, n)')


def _check_time_independent(x: TimeQArray, argname: str):
    # a time-dependent Lindbladian has no steady state in general, and a clipped
    # constant vanishes outside its time window
    if not (isinstance(x, ConstantTimeQArray) and _unclipped(x)):
        raise TypeError(
            f'Argument `{argname}` must be time-independent, but is a time-dependent '
            f'`{type(x).__name__}`.'
        )
