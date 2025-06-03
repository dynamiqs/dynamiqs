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
    times: ArrayLike | None = None,
    hlines: bool = True,
):
    r"""Plot the expectation value of the Pauli operators of a qubit $\sigma_x$,
    $\sigma_y$ and $\sigma_z$ as a function of time.

    Warning:
        Documentation redaction in progress.

    Examples:
        >>> H = dq.sigmax() + dq.sigmay()
        >>> jump_ops = [jnp.sqrt(0.2) * dq.sigmam()]
        >>> psi0 = dq.excited()
        >>> tsave = jnp.linspace(0, 10.0, 1001)
        >>> result = states = dq.mesolve(H, jump_ops, psi0, tsave)
        >>> dq.plot.xyz(result.states, times=tsave)
        >>> renderfig('plot_xyz')

        ![plot_xyz](../../figs_code/plot_xyz.png){.fig}
    """
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
