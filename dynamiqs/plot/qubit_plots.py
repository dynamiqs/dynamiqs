from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike
from matplotlib.axes import Axes

from .._checks import check_shape
from ..qarrays.qarray import QArrayLike
from ..qarrays.utils import to_jax
from ..utils.general import expect
from ..utils.operators import xyz as sigmaxyz
from .utils import optional_ax

__all__ = ['xyz']


@optional_ax
def xyz(
    states: QArrayLike,
    *,
    ax: Axes | None = None,
    w: float = 7.0,
    h: float | None = None,
    times: ArrayLike | None = None,
    hlines: bool = True,
):
    r"""Plot the expectation values of the Pauli operators of a qubit over time.

    The expectation values $\langle\sigma_x\rangle$, $\langle\sigma_y\rangle$ and
    $\langle\sigma_z\rangle$ are plotted for each state.

    Args:
        states (qarray-like of shape (N, 2, 1) or (N, 2, 2)): Sequence of _N_ qubit
            kets or density matrices.
        ax: Matplotlib `Axes` to plot on. If `None`, a new figure of size `(w, h)`
            is created.
        w: Width of the new figure in inches, ignored if `ax` is specified.
        h: Height of the new figure in inches, ignored if `ax` is specified. If
            `None`, defaults to `w / 1.6`.
        times (array-like of shape (N,), optional): Times of the states, used for
            the x axis. If `None`, the x axis is the index of the state.
        hlines: Whether to draw dashed horizontal lines at $\pm1$.

    Examples:
        >>> H = dq.sigmax() + dq.sigmay()
        >>> jump_ops = [jnp.sqrt(0.2) * dq.sigmam()]
        >>> psi0 = dq.excited()
        >>> tsave = jnp.linspace(0, 10.0, 1001)
        >>> result = dq.mesolve(H, jump_ops, psi0, tsave)
        >>> dq.plot.xyz(result.states, times=tsave)
        >>> renderfig('plot_xyz')

        ![plot_xyz](../../figs_code/plot_xyz.png){.fig}
    """
    assert ax is not None

    states = to_jax(states)
    times = jnp.asarray(times) if times is not None else None
    check_shape(states, 'states', '(N, n, 1)', '(N, n, n)')

    x = jnp.arange(len(states)) if times is None else times
    y = expect(sigmaxyz(), states).real  # (3, nstates)
    y = y.T  # (nstates, 3)
    label = [rf'$\langle \sigma_{k}\rangle$' for k in ['x', 'y', 'z']]
    ax.plot(x, y, label=label)
    ax.set(ylim=(-1.0 - 0.06, 1.0 + 0.06))
    ax.legend()

    if hlines:
        ax.axhline(-1.0, ls='--', color='gray', lw=1.0)
        ax.axhline(1.0, ls='--', color='gray', lw=1.0)
