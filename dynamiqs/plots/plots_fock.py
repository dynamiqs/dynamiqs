from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, LogNorm, Normalize

from .._checks import check_shape, check_times
from ..utils.utils import isdm, isket
from .utils import (
    add_colorbar,
    colors,
    integer_ticks,
    ket_ticks,
    optional_ax,
    sample_cmap,
)

__all__ = ['plot_fock', 'plot_fock_evolution']


def _populations(x: ArrayLike) -> Array:
    x = jnp.asarray(x)
    if isket(x):
        return jnp.abs(x.squeeze(-1)) ** 2
    elif isdm(x):
        # batched extract diagonal
        bdiag = jnp.vectorize(jnp.diag, signature='(a,b)->(c)')
        return bdiag(x).real
    else:
        raise TypeError


@optional_ax
def plot_fock(
    state: ArrayLike,
    *,
    ax: Axes | None = None,
    allxticks: bool = True,
    ymax: float | None = 1.0,
    color: str = colors['blue'],
    alpha: float = 1.0,
):
    """Plot the photon number population of a state.

    Warning:
        Documentation redaction in progress.

    Examples:
        >>> psi = dq.coherent(16, 2.0)
        >>> dq.plot_fock(psi)
        >>> renderfig('plot_fock')

        ![plot_fock](/figs-code/plot_fock.png){.fig}

        >>> # the even cat state has only even photon number components
        >>> psi = dq.unit(dq.coherent(32, 3.0) + dq.coherent(32, -3.0))
        >>> dq.plot_fock(psi, allxticks=False, ymax=None)
        >>> renderfig('plot_fock_even_cat')

        ![plot_fock_even_cat](/figs-code/plot_fock_even_cat.png){.fig}

        >>> dq.plot_fock(dq.coherent(16, 1.0), alpha=0.5)
        >>> dq.plot_fock(dq.coherent(16, 2.0), ax=plt.gca(), alpha=0.5, color='red')
        >>> renderfig('plot_fock_coherent')

        ![plot_fock_coherent](/figs-code/plot_fock_coherent.png){.fig}
    """
    state = jnp.asarray(state)
    check_shape(state, 'state', '(n, 1)', '(n, n)')

    n = state.shape[0]
    x = range(n)
    y = _populations(state)

    # plot
    ax.bar(x, y, color=color, alpha=alpha)
    if ymax is not None:
        ax.set_ylim(ymax=ymax)
    ax.set(xlim=(0 - 0.5, n - 0.5))

    # set x ticks
    integer_ticks(ax.xaxis, n, all=allxticks)
    ket_ticks(ax.xaxis)


@optional_ax
def plot_fock_evolution(
    states: ArrayLike,
    *,
    ax: Axes | None = None,
    times: ArrayLike | None = None,
    cmap: str = 'Blues',
    logscale: bool = False,
    logvmin: float = 1e-4,
    colorbar: bool = True,
    allyticks: bool = True,
):
    """Plot the photon number population of state as a function of time.

    Warning:
        Documentation redaction in progress.

    Examples:
        >>> n = 16
        >>> a = dq.destroy(n)
        >>> psi0 = dq.coherent(16, 0.0)
        >>> H = 2.0 * (a + dq.dag(a))
        >>> tsave = np.linspace(0, 1.0, 11)
        >>> result = dq.sesolve(H, psi0, tsave)
        >>> dq.plot_fock_evolution(result.states)
        >>> renderfig('plot_fock_evolution')

        ![plot_fock_evolution](/figs-code/plot_fock_evolution.png){.fig}

        Use the log scale option to visualise low populations:
        >>> dq.plot_fock_evolution(result.states, logscale=True, logvmin=1e-5)
        >>> renderfig('plot_fock_evolution_log')

        ![plot_fock_evolution_log](/figs-code/plot_fock_evolution_log.png){.fig}
    """
    states = jnp.asarray(states)
    times = jnp.asarray(times) if times is not None else None
    check_shape(states, 'states', '(N, n, 1)', '(N, n, n)')
    if times is not None:
        times = check_times(times, 'times')

    x = jnp.arange(len(states)) if times is None else times
    n = states[0].shape[0]
    y = range(n)
    z = _populations(states).T

    # set norm and colormap
    if logscale:
        norm = LogNorm(vmin=logvmin, vmax=1.0, clip=True)
        # stepped cmap
        ncolors = int(jnp.log10(1 / logvmin))
        clist = sample_cmap(cmap, ncolors + 2)[1:-1]  # remove extremal colors
        cmap = ListedColormap(clist)
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)

    # plot
    ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    ax.grid(False)

    # set y ticks
    integer_ticks(ax.yaxis, n, all=allyticks)
    ket_ticks(ax.yaxis)

    if colorbar:
        add_colorbar(ax, cmap, norm, size='2%', pad='2%')
