from __future__ import annotations

import warnings
from math import isclose

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from ..utils.tensor_types import ArrayLike, to_numpy, to_tensor
from ..utils.utils import norm, unit
from ..utils.wigners import wigner
from .utils import add_colorbar, colors, gridplot, linmap, optax

__all__ = ['plot_wigner', 'plot_wigner_mosaic']


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
    normalize: bool = True,
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
    if normalize:
        norm_state = norm(state).item()
        if not isclose(norm_state, 1.0, rel_tol=1e-4):
            warnings.warn(
                'The state has been normalized to compute the Wigner (expected norm to'
                f' be 1.0 but norm is {norm_state:.4f}).'
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
