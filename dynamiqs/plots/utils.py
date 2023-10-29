from __future__ import annotations

from functools import wraps

import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator, NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

__all__ = [
    'figax',
    'optax',
    'mplstyle',
    'integer_ticks',
    'sample_cmap',
    'minorticks_off',
    'fock_ticks',
    'add_colorbar',
]


def figax(w: float = 10.0, h: float | None = None, **kwargs) -> tuple[Figure, Axes]:
    """Return a figure with specified width and length."""
    if h is None:
        h = w / 2
    return plt.subplots(1, 1, figsize=(w, h), **kwargs)


def optax(func):
    """Decorator to build an `Axes` object to pass as an argument to a plot
    function if it wasn't passed by the user.

    Examples:
        Replace
        ```
        def myplot(ax=None):
            if ax is None:
                _, ax = plt.subplots(1, 1)

            # ...
        ```
        by
        ```
        @optax
        def myplot(ax=None):
            # ...
        ```
    """

    @wraps(func)
    def wrapper(
        *args,
        ax: Axes | None = None,
        w: float = 10.0,
        h: float | None = None,
        **kwargs,
    ):
        if ax is None:
            _, ax = figax(w=w, h=h)
        return func(*args, ax=ax, **kwargs)

    return wrapper


colors = {
    'blue': '#0c5dA5',
    'red': '#ff6b6b',
    'turquoise': '#2ec4b6',
    'yellow': '#ffc463',
    'grey': '#9e9e9e',
    'purple': '#845b97',
    'brown': '#c0675c',
    'darkgreen': '#20817d',
    'darkgrey': '#666666',
}


def mplstyle(*, latex: bool = True):
    """Set custom Matplotlib style."""
    plt.rcParams.update(
        {
            # xtick
            'xtick.direction': 'in',
            'xtick.major.size': 4.5,
            'xtick.minor.size': 2.5,
            'xtick.major.width': 1.0,
            'xtick.labelsize': 12,
            'xtick.minor.visible': True,
            # ytick
            'ytick.direction': 'in',
            'ytick.major.size': 4.5,
            'ytick.minor.size': 2.5,
            'ytick.major.width': 1.0,
            'ytick.labelsize': 12,
            'ytick.minor.visible': True,
            # axes
            'axes.facecolor': 'white',
            'axes.grid': False,
            'axes.titlesize': 12,
            'axes.labelsize': 12,
            'axes.linewidth': 1.0,
            'axes.prop_cycle': cycler('color', colors.values()),
            # grid
            'grid.color': 'gray',
            'grid.linestyle': '--',
            'grid.alpha': 0.3,
            # legend
            'legend.frameon': False,
            'legend.fontsize': 12,
            # figure
            'figure.facecolor': 'white',
            'figure.dpi': 72,
            # other
            'savefig.facecolor': 'white',
            'font.size': 12,
            'scatter.marker': 'x',
            'lines.linewidth': 2.0,
        }
    )
    if latex:
        plt.rcParams.update(
            {
                'text.usetex': latex,
                'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{braket}',
                'font.family': 'serif',
                'font.serif': 'Computer Modern Roman',
            }
        )


def integer_ticks(axis: Axis):
    # let maptlotlib choose major ticks position but restrict to integers
    axis.get_major_locator().set_params(integer=True)

    # format major ticks as integer
    axis.set_major_formatter(lambda x, _: f'{int(x)}')

    # fix minor ticks to integer positions only
    axis.set_minor_locator(MultipleLocator(1))


def fock_ticks(axis: Axis, n: int, all: bool = True):
    if all:
        axis.set_ticks(range(n))
        minorticks_off(axis)
    else:
        integer_ticks(axis)
    axis.set_major_formatter(lambda x, _: fr'$|{{{int(x)}}}\rangle$')


def sample_cmap(name: str, n: int, alpha: float = 1.0) -> np.ndarray:
    sampled_cmap = get_cmap(name)(np.linspace(0, 1, n))
    sampled_cmap[:, -1] = alpha
    return sampled_cmap


def minorticks_off(axis: Axis):
    axis.set_minor_locator(NullLocator())


def add_colorbar(
    ax: Axes, cmap: str, norm: Normalize, *, size: str = '5%', pad: str = '5%'
) -> Axes:
    # append a new axes on the right with the same height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=size, pad=pad)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(mappable=mappable, cax=cax)
    cax.grid(False)
    return cax
