from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, LogNorm, Normalize

from ..utils.tensor_types import ArrayLike, to_numpy
from .utils import add_colorbar, fock_ticks, optax, sample_cmap

__all__ = [
    'plot_pwc_pulse',
    'plot_fock',
    'plot_focks',
]


@optax
def plot_pwc_pulse(
    times: ArrayLike,
    values: ArrayLike,
    *,
    ax: Axes = None,
    ycenter: bool = True,
    real_color='#0C5DA5',
    imag_color='#845B97',
):
    times = to_numpy(times)
    values = to_numpy(values)

    # format times and values, for example:
    # times  = [0, 1, 2, 3] -> [0, 1, 1, 2, 2, 3, 3, 4]
    # values = [6, 7, 8, 9] -> [6, 6, 7, 7, 8, 8, 9, 9]
    times = times.repeat(2)[1:-1]
    values = values.repeat(2)
    zeros = np.zeros_like(times)

    # real part
    ax.plot(times, values.real, alpha=0.6, lw=2.0, label='real', color=real_color)
    ax.fill_between(times, zeros, values.real, alpha=0.2, color=real_color)

    # imaginary part
    ax.plot(times, values.imag, alpha=0.6, lw=2.0, label='imag', color=imag_color)
    ax.fill_between(times, zeros, values.imag, alpha=0.2, color=imag_color)

    ax.legend(loc='lower right')

    if ycenter:
        ymax = max(ax.get_ylim(), key=abs)
        ax.set_ylim(ymin=-ymax, ymax=ymax)


def populations(s: np.ndarray) -> np.ndarray:
    return np.abs(s.squeeze()) ** 2 if s.shape[1] == 1 else np.real(np.diag(s))


@optax
def plot_fock(
    state: ArrayLike,
    *,
    ax: Axes | None = None,
    alpha: float = 1.0,
    xticksall: bool = True,
):
    """Plot the photon number population."""
    state = to_numpy(state)
    n = state.shape[0]
    x = range(n)
    y = populations(state)

    # plot
    ax.bar(x, y, alpha=alpha)
    ax.set(xlim=(0 - 0.5, n - 0.5), ylim=(0, 1 + 0.05))

    # set x ticks
    fock_ticks(ax.xaxis, n, all=xticksall)


@optax
def plot_focks(
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
    """Plot the photon number population as a function of time."""
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
