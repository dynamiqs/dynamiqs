from __future__ import annotations

import pathlib
import shutil

import imageio as iio
import IPython.display as ipy
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from tqdm import tqdm

from .._checks import check_shape
from ..utils import wigner as compute_wigner
from .utils import add_colorbar, colors, figax, grid, optional_ax

__all__ = ['wigner', 'wigner_mosaic', 'wigner_gif']


@optional_ax
def plot_wigner_data(
    wigner: ArrayLike,
    xmax: float,
    ymax: float,
    *,
    ax: Axes | None = None,
    vmax: float = 2 / jnp.pi,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    colorbar: bool = True,
    cross: bool = False,
    clear: bool = False,
):
    w = jnp.asarray(wigner)
    check_shape(w, 'wigner', '(n, n)')

    # set plot norm
    vmin = -vmax
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    # clip to avoid rounding errors
    w = w.clip(vmin, vmax)

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
        if vmax == 2 / jnp.pi:
            cax.set_yticks([vmin, 0.0, vmax], labels=[r'$-2/\pi$', r'$0$', r'$2/\pi$'])

    if cross:
        ax.axhline(0.0, color=colors['grey'], ls='-', lw=0.7, alpha=0.8)
        ax.axvline(0.0, color=colors['grey'], ls='-', lw=0.7, alpha=0.8)

    if clear:
        ax.grid(False)
        ax.axis(False)


@optional_ax
def wigner(
    state: ArrayLike,
    *,
    ax: Axes | None = None,
    xmax: float = 5.0,
    ymax: float | None = None,
    vmax: float = 2 / jnp.pi,
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

    Examples:
        >>> psi = dq.coherent(16, 2.0)
        >>> dq.plot.wigner(psi)
        >>> renderfig('plot_wigner_coh')

        ![plot_wigner_coh](/figs_code/plot_wigner_coh.png){.fig-half}

        >>> psi = dq.unit(dq.coherent(16, 2) + dq.coherent(16, -2))
        >>> dq.plot.wigner(psi, xmax=4.0, ymax=2.0, colorbar=False)
        >>> renderfig('plot_wigner_cat')

        ![plot_wigner_cat](/figs_code/plot_wigner_cat.png){.fig-half}

        >>> psi = dq.unit(dq.fock(2, 0) + dq.fock(2, 1))
        >>> dq.plot.wigner(psi, xmax=2.0, cross=True)
        >>> renderfig('plot_wigner_01')

        ![plot_wigner_01](/figs_code/plot_wigner_01.png){.fig-half}

        >>> psi = dq.unit(sum(dq.coherent(32, 3 * a) for a in [1, 1j, -1, -1j]))
        >>> dq.plot.wigner(psi, npixels=201, clear=True)
        >>> renderfig('plot_wigner_4legged')

        ![plot_wigner_4legged](/figs_code/plot_wigner_4legged.png){.fig-half}
    """
    state = jnp.asarray(state)
    check_shape(state, 'state', '(n, 1)', '(n, n)')

    ymax = xmax if ymax is None else ymax
    _, _, w = compute_wigner(state, xmax, ymax, npixels)

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


def wigner_mosaic(
    states: ArrayLike,
    *,
    n: int = 8,
    nrows: int = 1,
    w: float = 3.0,
    h: float | None = None,
    xmax: float = 5.0,
    ymax: float | None = None,
    vmax: float = 2 / jnp.pi,
    npixels: int = 101,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    cross: bool = False,
):
    r"""Plot the Wigner function of multiple states in a mosaic arrangement.

    Warning:
        Documentation redaction in progress.

    See [`dq.plot.wigner()`][dynamiqs.plot.wigner] for more details.

    Examples:
        >>> psis = [dq.fock(3, i) for i in range(3)]
        >>> dq.plot.wigner_mosaic(psis)
        >>> renderfig('plot_wigner_mosaic_fock')

        ![plot_wigner_mosaic_fock](/figs_code/plot_wigner_mosaic_fock.png){.fig}

        >>> n = 16
        >>> a = dq.destroy(n)
        >>> H = dq.zero(n)
        >>> jump_ops = [a @ a - 4.0 * dq.eye(n)]  # cat state inflation
        >>> psi0 = dq.coherent(n, 0)
        >>> tsave = jnp.linspace(0, 1.0, 101)
        >>> result = dq.mesolve(H, jump_ops, psi0, tsave)
        >>> dq.plot.wigner_mosaic(result.states, n=6, xmax=4.0, ymax=2.0)
        >>> renderfig('plot_wigner_mosaic_cat')

        ![plot_wigner_mosaic_cat](/figs_code/plot_wigner_mosaic_cat.png){.fig}

        >>> n = 16
        >>> a = dq.destroy(n)
        >>> H = dq.dag(a) @ dq.dag(a) @ a @ a  # Kerr Hamiltonian
        >>> psi0 = dq.coherent(n, 2)
        >>> tsave = jnp.linspace(0, jnp.pi, 101)
        >>> result = dq.sesolve(H, psi0, tsave)
        >>> dq.plot.wigner_mosaic(result.states, n=25, nrows=5, xmax=4.0)
        >>> renderfig('plot_wigner_mosaic_kerr')

        ![plot_wigner_mosaic_kerr](/figs_code/plot_wigner_mosaic_kerr.png){.fig}
    """
    states = jnp.asarray(states)
    check_shape(states, 'states', '(N, n, 1)', '(N, n, n)')

    nstates = len(states)
    n = min(nstates, n)

    # create grid of plot
    _, axs = grid(
        n,
        nrows=nrows,
        w=w,
        h=h,
        gridspec_kw=dict(wspace=0, hspace=0),
        sharex=True,
        sharey=True,
    )

    ymax = xmax if ymax is None else ymax
    selected_indexes = np.linspace(0, nstates, n, dtype=int)
    _, _, wig = compute_wigner(states[selected_indexes], xmax, ymax, npixels)

    # plot individual wigner
    for i, ax in enumerate(axs):
        plot_wigner_data(
            wig[i],
            ax=ax,
            xmax=xmax,
            ymax=ymax,
            vmax=vmax,
            cmap=cmap,
            interpolation=interpolation,
            colorbar=False,
            cross=cross,
            clear=False,
        )
        ax.set(xlabel='', ylabel='', xticks=[], yticks=[])


def wigner_gif(
    states: ArrayLike,
    *,
    gif_duration: float = 5.0,
    fps: int = 10,
    w: float = 5.0,
    h: float | None = None,
    xmax: float = 5.0,
    ymax: float | None = None,
    vmax: float = 2 / jnp.pi,
    npixels: int = 101,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    cross: bool = False,
    clear: bool = False,
    filename: str = '.tmp/dynamiqs/wigner.gif',
    dpi: int = 72,
    display: bool = True,
):
    r"""Plot a GIF of the Wigner function of multiple states.

    Warning:
        Documentation redaction in progress.

    Warning:
        This function creates files in the current working directory under
        `.tmp/dynamiqs`.

    Note:
        By default, the GIF is displayed in Jupyter notebook environments.

    See [`dq.plot.wigner()`][dynamiqs.plot.wigner] for more details.

    Examples:
        >>> n = 16
        >>> a = dq.destroy(n)
        >>> H = dq.zero(n)
        >>> jump_ops = [a @ a - 4.0 * dq.eye(n)]  # cat state inflation
        >>> psi0 = dq.coherent(n, 0)
        >>> tsave = jnp.linspace(0, 1.0, 1001)
        >>> result = dq.mesolve(H, jump_ops, psi0, tsave)
        >>> dq.plot.wigner_gif(
        ...     result.states,
        ...     fps=25,  # 25 frames per second
        ...     xmax=4.0,
        ...     ymax=2.0,
        ...     filename='docs/figs_code/wigner-cat.gif',
        ...     dpi=150,
        ...     display=False,
        ... )

        ![plot_wigner_gif_cat](/figs_code/wigner-cat.gif){.fig}

        >>> n = 16
        >>> a = dq.destroy(n)
        >>> H = dq.dag(a) @ dq.dag(a) @ a @ a  # Kerr Hamiltonian
        >>> psi0 = dq.coherent(n, 2)
        >>> tsave = jnp.linspace(0, jnp.pi, 1001)
        >>> result = dq.sesolve(H, psi0, tsave)
        >>> dq.plot.wigner_gif(
        ...     result.states,
        ...     gif_duration=10.0,  # 10 seconds GIF
        ...     fps=25,
        ...     xmax=4.0,
        ...     clear=True,
        ...     filename='docs/figs_code/wigner-kerr.gif',
        ...     dpi=150,
        ...     display=False,
        ... )

        ![plot_wigner_gif_kerr](/figs_code/wigner-kerr.gif){.fig-half}
    """
    states = jnp.asarray(states)
    check_shape(states, 'states', '(N, n, 1)', '(N, n, n)')

    ymax = xmax if ymax is None else ymax
    nframes = int(gif_duration * fps)
    selected_indexes = np.linspace(0, len(states), nframes, dtype=int)
    _, _, wig = compute_wigner(states[selected_indexes], xmax, ymax, npixels)

    try:
        # create temporary directory
        tmpdir = pathlib.Path('./.tmp/dynamiqs')
        tmpdir.mkdir(parents=True, exist_ok=True)

        frames = []
        for i in tqdm(range(nframes)):
            fig, ax = figax(w=w, h=h)

            plot_wigner_data(
                wig[i],
                ax=ax,
                xmax=xmax,
                ymax=ymax,
                vmax=vmax,
                cmap=cmap,
                interpolation=interpolation,
                colorbar=False,
                cross=cross,
                clear=clear,
            )

            frame_filename = tmpdir / f'tmp-{i}.png'
            fig.savefig(frame_filename, bbox_inches='tight', dpi=dpi)
            plt.close()
            frame = iio.v3.imread(frame_filename)
            frames.append(frame)

        # loop=0: loop the GIF forever
        frame_duration_ms = 1000 * 1 / fps
        iio.v3.imwrite(
            filename, frames, format='GIF', duration=frame_duration_ms, loop=0
        )
        if display:
            ipy.display(ipy.Image(filename))
    finally:
        if tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)
