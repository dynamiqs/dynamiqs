from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, LogNorm, Normalize

from ..utils.tensor_types import ArrayLike, to_numpy
from .utils import add_colorbar, colors, integer_ticks, ket_ticks, optax, sample_cmap

__all__ = ['plot_fock', 'plot_fock_evolution']


def populations(s: np.ndarray) -> np.ndarray:
    return np.abs(s.squeeze()) ** 2 if s.shape[1] == 1 else np.real(np.diag(s))


@optax
def plot_fock(
    state: ArrayLike,
    *,
    ax: Axes | None = None,
    allxticks: bool = True,
    ymax: float | None = 1.0,
    color: str = colors['blue'],
):
    """Plot the photon number population of a state.

    Warning:
        Documentation redaction in progress.

    Examples:
        >>> psi = dq.coherent(16, 2.0)
        >>> dq.plot_fock(psi)
        >>> renderfig('plot_fock')

        ![plot_fock](/figs-code/plot_fock.png){.fig}
    """
    state = to_numpy(state)
    n = state.shape[0]
    x = range(n)
    y = populations(state)

    # plot
    ax.bar(x, y, color=color)
    if ymax is not None:
        ax.set_ylim(ymax=ymax)
    ax.set(xlim=(0 - 0.5, n - 0.5))

    # set x ticks
    integer_ticks(ax.xaxis, n, all=allxticks)
    ket_ticks(ax.xaxis)


@optax
def plot_fock_evolution(
    states: list[ArrayLike],
    *,
    ax: Axes | None = None,
    times: np.ndarray | None = None,
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
        >>> H = 2.0 * (a + a.mH)
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
    states = to_numpy(states)

    x = np.arange(len(states)) if times is None else times
    n = states[0].shape[0]
    y = range(n)
    z = np.array([populations(s) for s in states]).T

    # set norm and colormap
    if logscale:
        norm = LogNorm(vmin=logvmin, vmax=1.0, clip=True)
        # stepped cmap
        ncolors = int(np.log10(1 / logvmin))
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
