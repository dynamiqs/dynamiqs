from __future__ import annotations

import warnings
from math import isclose

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, LogNorm, Normalize

from ..utils.tensor_types import ArrayLike, to_numpy, to_tensor
from ..utils.utils import norm, unit
from ..utils.wigners import wigner
from .utils import (
    add_colorbar,
    colors,
    fock_ticks,
    gridplot,
    linmap,
    optax,
    sample_cmap,
)

__all__ = [
    'plot_wigner',
    'plot_wigner_mosaic',
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
    vmax: float = 2 / np.pi,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    colorbar: bool = True,
    cross: bool = False,
    clear: bool = False,
):
    w = to_numpy(wigner)

    # set plot norm
    vmin = -vmax
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    # clip to avoid rounding errors
    w = np.clip(w, vmin, vmax)

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
        if vmax == 2 / np.pi:
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
    xmax: float = 5.0,
    ymax: float | None = None,
    vmax: float = 2 / np.pi,
    npixels: int = 101,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    colorbar: bool = True,
    cross: bool = False,
    clear: bool = False,
):
    r"""Plot the Wigner function of a state.

    Warning:
        Documentation redaction in progress.

    Note:
        Choose a diverging colormap `cmap` for better results.

    Warning:
        The axis scaling is chosen so that a coherent state $\ket{\alpha}$ lies at the
        coordinates $(x,y)=(\mathrm{Re}(\alpha),\mathrm{Im}(\alpha))$, which is
        different from the default behaviour of `qutip.plot_wigner()`.

    Warning-: Non-normalized state
        If the given state is not normalized, it will be normalized before plotting
        and a warning will be issued. If you want to ignore the warning, use
        ```python
        import warnings
        warnings.filterwarnings('ignore', module='dynamiqs')
        ```

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
        >>> dq.plot_wigner(psi, xmax=2.0, cross=True)
        >>> renderfig('plot_wigner_01')

        ![plot_wigner_01](/figs-code/plot_wigner_01.png){.fig-half}

        >>> psi = dq.unit(sum(dq.coherent(32, 3 * a) for a in [1, 1j, -1, -1j]))
        >>> dq.plot_wigner(psi, npixels=201, clear=True)
        >>> renderfig('plot_wigner_4legged')

        ![plot_wigner_4legged](/figs-code/plot_wigner_4legged.png){.fig-half}
    """
    state = to_tensor(state)

    # normalize state
    norm_state = norm(state).item()
    if not isclose(norm_state, 1.0, rel_tol=1e-4):
        warnings.warn(
            'The state has been normalized to compute the Wigner (expected norm to be'
            f' 1.0 but norm is {norm_state:.4f}).'
        )
        state = unit(state)

    ymax = xmax if ymax is None else ymax

    _, _, w = wigner(state, xmax=xmax, ymax=ymax, npixels=npixels)

    plot_wigner_data(
        w,
        xmax,
        ymax,
        ax=ax,
        vmax=vmax,
        cmap=cmap,
        interpolation=interpolation,
        colorbar=colorbar,
        cross=cross,
        clear=clear,
    )


def plot_wigner_mosaic(
    states: ArrayLike,
    *,
    n: int = 8,
    nrows: int = 1,
    w: float = 3.0,
    h: float | None = None,
    xmax: float = 5.0,
    ymax: float | None = None,
    vmax: float = 2 / np.pi,
    npixels: int = 101,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    cross: bool = False,
):
    r"""Plot the Wigner function of multiple states in a mosaic arrangement.

    Warning:
        Documentation redaction in progress.

    See [`dq.plot_wigner()`][dynamiqs.plot_wigner] for more details.

    Examples:
        >>> psis = [dq.fock(3, i) for i in range(3)]
        >>> dq.plot_wigner_mosaic(psis)
        >>> renderfig('plot_wigner_mosaic_fock')

        ![plot_wigner_mosaic_fock](/figs-code/plot_wigner_mosaic_fock.png){.fig}

        >>> n = 16
        >>> a = dq.destroy(n)
        >>> H = dq.zero(n)
        >>> jump_ops = [a @ a - 4.0 * dq.eye(n)]
        >>> psi0 = dq.coherent(n, 0)
        >>> tsave = np.linspace(0, 1.0, 101)
        >>> result = dq.mesolve(H, jump_ops, psi0, tsave)
        >>> dq.plot_wigner_mosaic(result.states, n=6, xmax=4.0, ymax=2.0)
        >>> renderfig('plot_wigner_mosaic_cat')

        ![plot_wigner_mosaic_cat](/figs-code/plot_wigner_mosaic_cat.png){.fig}

        >>> n = 16
        >>> a = dq.destroy(n)
        >>> H = a.mH @ a.mH @ a @ a  # Kerr Hamiltonian
        >>> psi0 = dq.coherent(n, 2)
        >>> tsave = np.linspace(0, np.pi, 101)
        >>> result = dq.sesolve(H, psi0, tsave)
        >>> dq.plot_wigner_mosaic(result.states, n=25, nrows=5, xmax=4.0)
        >>> renderfig('plot_wigner_mosaic_kerr')

        ![plot_wigner_mosaic_kerr](/figs-code/plot_wigner_mosaic_kerr.png){.fig}
    """
    states = to_tensor(states)

    nstates = len(states)
    if nstates < n:
        n = nstates

    # todo: precompute batched wigners

    # create grid of plot
    _, axs = gridplot(
        n,
        nrows=nrows,
        w=w,
        h=h,
        gridspec_kw=dict(wspace=0, hspace=0),
        sharex=True,
        sharey=True,
    )

    # individual wigner plot options
    kwargs = dict(
        xmax=xmax,
        ymax=ymax,
        vmax=vmax,
        npixels=npixels,
        cmap=cmap,
        interpolation=interpolation,
        colorbar=False,
        cross=cross,
        clear=False,
    )

    # plot individual wigner
    for i in range(n):
        ax = next(axs)
        idx = int(linmap(i, 0, n - 1, 0, nstates - 1))
        plot_wigner(states[idx], ax=ax, **kwargs)
        ax.set(xlabel='', ylabel='', xticks=[], yticks=[])


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
