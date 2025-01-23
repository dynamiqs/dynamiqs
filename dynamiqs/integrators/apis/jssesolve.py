from __future__ import annotations

from jaxtyping import ArrayLike, PRNGKeyArray

from ...gradient import Gradient
from ...options import Options
from ...qarrays.qarray import QArrayLike
from ...result import JSSESolveResult
from ...solver import Solver
from ...time_qarray import TimeQArray


def jssesolve(
    H: QArrayLike | TimeQArray,
    jump_ops: list[QArrayLike | TimeQArray],
    psi0: QArrayLike,
    tsave: ArrayLike,
    keys: PRNGKeyArray,
    *,
    exp_ops: list[QArrayLike] | None = None,
    solver: Solver | None = None,
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> JSSESolveResult:
    r"""Solve the jump stochastic Schrödinger equation (SSE).

    Warning:
        This function has not been implemented yet. The following API is indicative
        of the planned implementation.

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

    Warning:
        For now, `jssesolve()` only supports linearly spaced `tsave` with values that
        are exact multiples of the solver fixed step size `dt`.

    Args:
        H _(qarray-like or time-qarray of shape (...H, n, n))_: Hamiltonian.
        jump_ops _(list of qarray-like or time-qarray, each of shape (n, n))_: List of
            jump operators.
        psi0 _(qarray-like of shape (...psi0, n, 1))_: Initial state.
        tsave _(array-like of shape (ntsave,))_: Times at which the states and
            expectation values are saved. The equation is solved from `tsave[0]` to
            `tsave[-1]`.
        keys _(list of PRNG keys)_: PRNG keys used to sample the point processes.
            The number of elements defines the number of sampled stochastic
            trajectories.
        exp_ops _(list of array-like, each of shape (n, n), optional)_: List of
            operators for which the expectation value is computed.
        solver: Solver for the integration. No defaults for now, you have to specify a
            solver (supported: [`EulerMaruyama`][dynamiqs.solver.EulerMaruyama]).
        gradient: Algorithm used to compute the gradient. The default is
            solver-dependent, refer to the documentation of the chosen solver for more
            details.
        options: Generic options (supported: `save_states`, `cartesian_batching`,
            `save_extra`, `nmaxclick`, `smart_sampling`).
            ??? "Detailed options API"
                ```
                dq.Options(
                    save_states: bool = True,
                    cartesian_batching: bool = True,
                    save_extra: callable[[Array], PyTree] | None = None,
                    nmaxclick: int = 10_000,
                    smart_sampling: bool = False,
                )
                ```

                **Parameters**

                - **save_states** - If `True`, the state is saved at every time in
                    `tsave`, otherwise only the final state is returned.
                - **cartesian_batching** - If `True`, batched arguments are treated as
                    separated batch dimensions, otherwise the batching is performed over
                    a single shared batched dimension.
                - **save_extra** _(function, optional)_ - A function with signature
                    `f(QArray) -> PyTree` that takes a state as input and returns a
                    PyTree. This can be used to save additional arbitrary data
                    during the integration, accessible in `result.extra`.
                - **nmaxclick** - Maximum buffer size for `result.clicktimes`, should be
                    set higher than the expected maximum number of jump event.
                - **smart_sampling** - If `True`, the no jump trajectory is simulated
                    only once, and only trajectories with one or more jumps are sampled
                    in `result.states`. The no jump state is accessible in
                    `result.no_jump_state` with its associated probability
                    `result.no_jump_proba`.

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
                    `None` up to `nmaxclick`.
                - **no_jump_state** _(..., nsave, n, 1)_ - Saved state for the no jump
                    trajectory, only if `options.smart_sampling=True`.
                - **no_jump_proba** _(..., nsave)_ - Probability of the no jump
                    trajectory, only if `options.smart_sampling=True`.
                - **extra** _(PyTree or None)_ - Extra data saved with `save_extra()` if
                    specified in `options`.
                - **keys** _(PRNG key array of shape (ntrajs,))_ - PRNG keys used to
                    sample the Wiener processes.
                - **infos** _(PyTree or None)_ - Solver-dependent information on the
                    resolution.
                - **tsave** _(array of shape (ntsave,))_ - Times for which results were
                    saved.
                - **solver** _(Solver)_ - Solver used.
                - **gradient** _(Gradient)_ - Gradient used.
                - **options** _(Options)_ - Options used.

    # Advanced use-cases

    ## Defining a time-dependent Hamiltonian or jump operator

    If the Hamiltonian or the jump operators depend on time, they can be converted
    to time-arrays using [`dq.pwc()`][dynamiqs.pwc],
    [`dq.modulated()`][dynamiqs.modulated], or
    [`dq.timecallable()`][dynamiqs.timecallable]. See the
    [Time-dependent operators](../../documentation/basics/time-dependent-operators.md)
    tutorial for more details.

    ## Running multiple simulations concurrently

    The Hamiltonian `H` and the initial state `psi0` can be batched to
    solve multiple SSEs concurrently. All other arguments (including the PRNG key)
    are common to every batch. The resulting states, click times and expectation values
    are batched according to the leading dimensions of `H` and `psi0`. The
    behaviour depends on the value of the `cartesian_batching` option.

    === "If `cartesian_batching = True` (default value)"
        The results leading dimensions are
        ```
        ... = ...H, ...psi0
        ```
        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `psi0` has shape _(4, n, 1)_,

        then `result.states` has shape _(2, 3, 4, ntrajs, ntsave, n, 1)_.

    === "If `cartesian_batching = False`"
        The results leading dimensions are
        ```
        ... = ...H = ...psi0  # (once broadcasted)
        ```
        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `psi0` has shape _(3, n, 1)_,

        then `result.states` has shape _(2, 3, ntrajs, ntsave, n, 1)_.

    See the
    [Batching simulations](../../documentation/basics/batching-simulations.md)
    tutorial for more details.

    Warning:
        Batching on `jump_ops` is not yet supported, if this is needed don't
        hesitate to
        [open an issue on GitHub](https://github.com/dynamiqs/dynamiqs/issues/new).
    """  # noqa: E501
    return NotImplemented
