from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, LogNorm, Normalize

from ..utils.tensor_types import ArrayLike, to_numpy
from .utils import add_colorbar, colors, fock_ticks, optax, sample_cmap

__all__ = [
    'plot_pwc_pulse',
    'plot_fock',
    'plot_fock_evolution',
]


@optax
def plot_pwc_pulse(
    times: ArrayLike,
    values: ArrayLike,
    *,
    ax: Axes = None,
    ycenter: bool = True,
    real_color: str = colors['blue'],
    imag_color: str = colors['purple'],
):
    """Plot a piecewise-constant pulse.

    Warning:
        Documentation redaction in progress.

    Examples:
        >>> n = 20
        >>> times = np.linspace(0, 1.0, n+1)
        >>> values = dq.rand_complex(n, seed=42)
        >>> dq.plot_pwc_pulse(times, values)
        >>> render('plot_pwc_pulse')

        ![plot_pwc_pulse](/figs-code/plot_pwc_pulse.png){.center}
    """
    times = to_numpy(times)  # (n + 1)
    values = to_numpy(values)  # (n)

    # format times and values, for example:
    # times  = [0, 1, 2, 3] -> [0, 1, 1, 2, 2, 3]
    # values = [4, 5, 6]    -> [4, 4, 5, 5, 6, 6]
    times = times.repeat(2)[1:-1]  # (2n)
    values = values.repeat(2)  # (2n)

    # real part
    ax.plot(times, values.real, label='real', color=real_color, alpha=0.7)
    ax.fill_between(times, 0, values.real, color=real_color, alpha=0.2)

    # imaginary part
    ax.plot(times, values.imag, label='imag', color=imag_color, alpha=0.7)
    ax.fill_between(times, 0, values.imag, color=imag_color, alpha=0.2)

    ax.legend(loc='lower right')

    if ycenter:
        ymax = max(ax.get_ylim(), key=abs)
        ax.set_ylim(ymin=-ymax, ymax=ymax)

    ax.set(xlim=(0, times[-1]))


def populations(s: np.ndarray) -> np.ndarray:
    return np.abs(s.squeeze()) ** 2 if s.shape[1] == 1 else np.real(np.diag(s))


@optax
def plot_fock(
    state: ArrayLike,
    *,
    ax: Axes | None = None,
    xticksall: bool = True,
    ymax: float | None = 1.0,
    color: str = colors['blue'],
):
    """Plot the photon number population of a state.

    Warning:
        Documentation redaction in progress.

    Examples:
        >>> psi = dq.coherent(16, 2.0)
        >>> dq.plot_fock(psi)
        >>> render('plot_fock')

        ![plot_fock](/figs-code/plot_fock.png){.center}
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
    fock_ticks(ax.xaxis, n, all=xticksall)


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
    yticksall: bool = True,
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
        >>> states = dq.sesolve(H, psi0, tsave).states
        >>> dq.plot_fock_evolution(states)
        >>> render('plot_fock_evolution')

        ![plot_fock_evolution](/figs-code/plot_fock_evolution.png){.center}

        Use the log scale option to visualise low populations:
        >>> dq.plot_fock_evolution(states, logscale=True, logvmin=1e-5)
        >>> render('plot_fock_evolution_log')

        ![plot_fock_evolution_log](/figs-code/plot_fock_evolution_log.png){.center}
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
    fock_ticks(ax.yaxis, n, all=yticksall)

    if colorbar:
        add_colorbar(ax, cmap, norm, size='2%', pad='2%')
