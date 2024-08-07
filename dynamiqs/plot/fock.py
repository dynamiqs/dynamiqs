from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, LogNorm, Normalize

from .._checks import check_shape, check_times
from ..utils.quantum_utils import isdm, isket
from .utils import (
    add_colorbar,
    colors,
    integer_ticks,
    ket_ticks,
    optional_ax,
    sample_cmap,
)

__all__ = ['fock', 'fock_evolution']


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
def fock(
    state: ArrayLike,
    *,
    ax: Axes | None = None,
    allxticks: bool = False,
    ymax: float | None = 1.0,
    color: str = colors['blue'],
    alpha: float = 1.0,
    label: str = '',
):
    """Plot the photon number population of a state.

    Warning:
        Documentation redaction in progress.

    Examples:
        >>> psi = dq.coherent(16, 2.0)
        >>> dq.plot.fock(psi)
        >>> renderfig('plot_fock')

        ![plot_fock](/figs_code/plot_fock.png){.fig}

        >>> # the even cat state has only even photon number components
        >>> psi = dq.unit(dq.coherent(32, 3.0) + dq.coherent(32, -3.0))
        >>> dq.plot.fock(psi, allxticks=False, ymax=None)
        >>> renderfig('plot_fock_even_cat')

        ![plot_fock_even_cat](/figs_code/plot_fock_even_cat.png){.fig}

        >>> dq.plot.fock(dq.coherent(16, 1.0), alpha=0.5)
        >>> dq.plot.fock(dq.coherent(16, 2.0), ax=plt.gca(), alpha=0.5, color='red')
        >>> renderfig('plot_fock_coherent')

        ![plot_fock_coherent](/figs_code/plot_fock_coherent.png){.fig}
    """
    state = jnp.asarray(state)
    check_shape(state, 'state', '(n, 1)', '(n, n)')

    n = state.shape[0]
    x = range(n)
    y = _populations(state)

    # plot
    ax.bar(x, y, color=color, alpha=alpha, label=label)
    if ymax is not None:
        ax.set_ylim(ymax=ymax)
    ax.set(xlim=(0 - 0.5, n - 0.5))

    # set x ticks
    integer_ticks(ax.xaxis, n, all=allxticks)
    ket_ticks(ax.xaxis)

    # turn legend on
    if label != '':
        ax.legend()


@optional_ax
def fock_evolution(
    states: ArrayLike,
    *,
    ax: Axes | None = None,
    times: ArrayLike | None = None,
    cmap: str = 'Blues',
    logscale: bool = False,
    logvmin: float = 1e-4,
    colorbar: bool = True,
    allyticks: bool = False,
):
    """Plot the photon number population of state as a function of time.

    Warning:
        Documentation redaction in progress.

    Examples:
        >>> n = 16
        >>> a = dq.destroy(n)
        >>> psi0 = dq.coherent(n, 0.0)
        >>> H = 2.0 * (a + dq.dag(a))
        >>> tsave = jnp.linspace(0, 1.0, 11)
        >>> result = dq.sesolve(H, psi0, tsave)
        >>> dq.plot.fock_evolution(result.states, times=tsave)
        >>> renderfig('plot_fock_evolution')

        ![plot_fock_evolution](/figs_code/plot_fock_evolution.png){.fig}

        Use the log scale option to visualise low populations:
        >>> dq.plot.fock_evolution(result.states, times=tsave, logscale=True)
        >>> renderfig('plot_fock_evolution_log')

        ![plot_fock_evolution_log](/figs_code/plot_fock_evolution_log.png){.fig}
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
        ncolors = jnp.round(jnp.log10(1 / logvmin)).astype(int)
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
