# ruff: noqa: ARG001

from __future__ import annotations

from jaxtyping import ArrayLike

from ...gradient import Gradient
from ...options import Options
from ...result import Result
from ...solver import Solver
from ...time_array import TimeArray


def smesolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    etas: ArrayLike,
    rho0: ArrayLike,
    tsave: ArrayLike,
    *,
    tmeas: ArrayLike | None = None,
    ntrajs: int = 10,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver | None = None,
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> Result:
    r"""Solve the diffusive stochastic master equation (SME).

    Warning:
        This function has not been ported to JAX yet. The following documentation is
        a draft API, copied from the old PyTorch version of the library.

    This function computes the evolution of the density matrix $\rho(t)$ at time $t$,
    starting from an initial state $\rho_0$, according to the diffusive SME in Itô
    form (with $\hbar=1$ and where time is implicit(1))
    $$
        \begin{split}
            \dd\rho =&~ -i[H, \rho]\,\dt + \sum_{k=1}^N \left(
                L_k \rho L_k^\dag
                - \frac{1}{2} L_k^\dag L_k \rho
                - \frac{1}{2} \rho L_k^\dag L_k
        \right)\dt \\\\
            &+ \sum_{k=1}^N \sqrt{\eta_k} \left(
                L_k \rho
                + \rho L_k^\dag
                - \tr{(L_k+L_k^\dag)\rho}\rho
            \right)\dd W_k,
        \end{split}
    $$
    where $H$ is the system's Hamiltonian, $\{L_k\}$ is a collection of jump operators,
    each continuously measured with efficiency $0\leq\eta_k\leq1$ ($\eta_k=0$ for
    purely dissipative loss channels) and $\dd W_k$ are independent Wiener processes.
    { .annotate }

    1. With explicit time dependence:
        - $\rho\to\rho(t)$
        - $H\to H(t)$
        - $L_k\to L_k(t)$
        - $\dd W_k\to \dd W_k(t)$

    Note-: Diffusive vs. jump SME
        In quantum optics the _diffusive_ SME corresponds to homodyne or heterodyne
        detection schemes, as opposed to the _jump_ SME which corresponds to photon
        counting schemes. No solver for the jump SME is provided yet, if this is needed
        don't hesitate to
        [open an issue on GitHub](https://github.com/dynamiqs/dynamiqs/issues/new).

    The measured signals $I_k=\dd y_k/\dt$ verifies (again time is implicit):
    $$
        \dd y_k =\sqrt{\eta_k} \tr{(L_k + L_k^\dag) \rho} \dt + \dd W_k.
    $$

    Note-: Signal normalisation
        Sometimes the signals are defined with a different but equivalent normalisation
        $\dd y_k' = \dd y_k/(2\sqrt{\eta_k})$.

    The signals $I_k$ are singular quantities, the solver returns the averaged signals
    $J_k$ defined for a time interval $[t_0, t_1)$ by:
    $$
        J_k([t_0, t_1)) = \frac{1}{t_1-t_0}\int_{t_0}^{t_1} I_k(t)\, \dt
        = \frac{1}{t_1-t_0}\int_{t_0}^{t_1} \dd y_k(t).
    $$
    The time intervals for integration are defined by the argument `tmeas`, which
    defines `len(tmeas) - 1` intervals. By default, `tmeas = tsave`, so the signals
    are averaged between the times at which the states are saved.

    Note-: Defining a time-dependent Hamiltonian or jump operator
        If the Hamiltonian or the jump operators depend on time, they can be converted
        to time-arrays using [`dq.pwc()`][dynamiqs.pwc],
        [`dq.modulated()`][dynamiqs.modulated], or
        [`dq.timecallable()`][dynamiqs.timecallable]. See the
        [Time-dependent operators](../../documentation/basics/time-dependent-operators.md)
        tutorial for more details.

    Note-: Running multiple simulations concurrently
        The Hamiltonian `H`, the jump operators `jump_ops` and the initial density
        matrix `rho0` can be batched to solve multiple SMEs concurrently. All other
        arguments are common to every batch. See the
        [Batching simulations](../../documentation/basics/batching-simulations.md)
        tutorial for
        more details.

    Args:
        H _(array-like or time-array of shape (bH?, n, n))_: Hamiltonian.
        jump_ops _(list of array-like or time-array, of shape (nL, n, n))_: List of
            jump operators.
        etas _(array-like of shape (nL,))_: Measurement efficiencies, must be of the
            same length as `jump_ops` with values between 0 and 1. For a purely
            dissipative loss channel, set the corresponding efficiency to 0. No
            measurement signal will be returned for such channels.
        rho0 _(array-like of shape (brho?, n, 1) or (brho?, n, n))_: Initial state.
        tsave _(array-like of shape (ntsave,))_: Times at which the states and
            expectation values are saved. The equation is solved from `tsave[0]` to
            `tsave[-1]`, or from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        tmeas _(array-like of shape (ntmeas,), optional)_: Times between which
            measurement signals are averaged and saved. Defaults to `tsave`.
        ntrajs: Number of stochastic trajectories to solve concurrently.
        exp_ops _(list of array-like, of shape (nE, n, n), optional)_: List of
            operators for which the expectation value is computed.
        solver: Solver for the integration.
        gradient: Algorithm used to compute the gradient. The default is
            solver-dependent, refer to the documentation of the chosen solver for more
            details.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].
    """  # noqa: E501
    return NotImplementedError
