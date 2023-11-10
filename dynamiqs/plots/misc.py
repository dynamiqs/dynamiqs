from __future__ import annotations

import numpy as np
import qutip as qt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, LogNorm, Normalize

from ..utils.tensor_types import ArrayLike, to_numpy, to_qutip, to_tensor
from .utils import add_colorbar, colors, fock_ticks, optax, sample_cmap

__all__ = [
    'plot_wigner',
    'plot_pwc_pulse',
    'plot_fock',
    'plot_fock_evolution',
]


@optax
def plot_wigner_data(
    wigner: ArrayLike,
    xmax: float,
    ymax: float,
    *,
    ax: Axes | None = None,
    cmap: str = 'RdBu',
    interpolation: str = 'bilinear',
    colorbar: bool = True,
    cross: bool = False,
    clear: bool = False,
):
    w = to_numpy(wigner)

    # set plot norm
    vmin, vmax = -2 / np.pi, 2 / np.pi
    norm = Normalize(vmin=vmin, vmax=vmax)

    # set the color of values outside of range [vmin, vmax]
    cmap = plt.get_cmap(cmap)
    cmap.set_over('black')
    cmap.set_under('black')

    # plot
    ax.imshow(
        w,
        cmap=cmap,
        norm=norm,
        origin='lower',
        aspect='equal',
        interpolation=interpolation,
        extent=[-xmax, xmax, -ymax, ymax],
    )

    # axis label
    ax.set(xlabel=r'$\mathrm{Re}(\alpha)$', ylabel=r'$\mathrm{Im}(\alpha)$')

    if colorbar and not clear:
        cax = add_colorbar(ax, cmap, norm)
        cax.set_yticks([vmin, 0.0, vmax], labels=[r'$-2/\pi$', r'$0$', r'$2/\pi$'])

    if cross:
        ax.axhline(0.0, color=colors['grey'], ls='-', lw=0.7, alpha=0.8)
        ax.axvline(0.0, color=colors['grey'], ls='-', lw=0.7, alpha=0.8)

    if clear:
        ax.grid(False)
        ax.axis(False)


@optax
def plot_wigner(
    state: ArrayLike,
    *,
    ax: Axes | None = None,
    lim: float = 5.0,
    xmax: float | None = None,
    ymax: float | None = None,
    npixels: int = 101,
    cmap: str = 'RdBu',
    interpolation: str = 'bilinear',
    colorbar: bool = True,
    cross: bool = False,
    clear: bool = False,
):
    r"""Plot the Wigner quasiprobability distribution of a state.

    Warning:
        Documentation redaction in progress.

    Note:
        Choose a diverging colormap `cmap` for better results.

    Warning:
        The axis scaling is chosen so that a coherent state $\ket{\alpha}$ lies at the
        coordinates $(x,y)=(\mathrm{Re}(\alpha),\mathrm{Im}(\alpha))$, which is
        different from the default behaviour of `qutip.plot_wigner()`.

    Examples:
        >>> psi = dq.coherent(16, 2.0)
        >>> dq.plot_wigner(psi)
        >>> renderfig('plot_wigner_coh')

        ![plot_wigner_coh](/figs-code/plot_wigner_coh.png){.fig-half}

        >>> psi = dq.unit(dq.coherent(16, 2) + dq.coherent(16, -2))
        >>> dq.plot_wigner(psi, xmax=4.0, ymax=2.0, colorbar=False)
        >>> renderfig('plot_wigner_cat')

        ![plot_wigner_cat](/figs-code/plot_wigner_cat.png){.fig-half}

        >>> psi = dq.unit(dq.fock(2, 0) + dq.fock(2, 1))
        >>> dq.plot_wigner(psi, lim=1.5, cross=True)
        >>> renderfig('plot_wigner_01')

        ![plot_wigner_01](/figs-code/plot_wigner_01.png){.fig-half}

        >>> psi = dq.unit(sum(dq.coherent(32, 3 * a) for a in [1, 1j, -1, -1j]))
        >>> dq.plot_wigner(psi, npixels=201, clear=True)
        >>> renderfig('plot_wigner_4legged')

        ![plot_wigner_4legged](/figs-code/plot_wigner_4legged.png){.fig-half}
    """
    state = to_tensor(state)

    xmax = lim if xmax is None else xmax
    ymax = lim if ymax is None else ymax

    # todo to use dynamiqs wigner function:
    #   - the wigner value is wrong by a factor 2
    #   - no way to set g=2 to properly center coherent states
    #   - choosing xmax!=ymax results in an incorrect Wigner

    # _, _, w = wigner(state, xmax=xmax, ymax=ymax, npixels=npixels)

    xvec = np.linspace(-xmax, xmax, npixels)
    yvec = np.linspace(-ymax, ymax, npixels)
    w = qt.wigner(to_qutip(state), xvec, yvec, g=2)

    plot_wigner_data(
        w,
        xmax,
        ymax,
        ax=ax,
        cmap=cmap,
        interpolation=interpolation,
        colorbar=colorbar,
        cross=cross,
        clear=clear,
    )


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
        >>> renderfig('plot_pwc_pulse')

        ![plot_pwc_pulse](/figs-code/plot_pwc_pulse.png){.fig}
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
    fock_ticks(ax.yaxis, n, all=yticksall)

    if colorbar:
        add_colorbar(ax, cmap, norm, size='2%', pad='2%')
