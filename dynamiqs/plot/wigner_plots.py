from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from IPython.display import Image
from jaxtyping import ArrayLike
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from .._checks import check_shape
from ..qarrays.qarray import QArrayLike
from ..qarrays.utils import asqarray, to_jax
from ..utils import wigner as compute_wigner
from .utils import add_colorbar, gif_indices, gifit, grid, optional_ax

__all__ = ['wigner_data', 'wigner', 'wigner_gif', 'wigner_mosaic']


@optional_ax
def wigner_data(
    wigner: ArrayLike,
    xmax: float,
    ymax: float,
    *,
    ax: Axes | None = None,
    w: float = 7.0,
    h: float | None = None,
    vmax: float | None = None,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    colorbar: bool = True,
    cross: bool = False,
    clear: bool = False,
    hbar: float = 0.5,
):
    r"""Plot a pre-computed Wigner function.

    Args:
        wigner (array-like of shape (npixels, npixels)): Wigner function on a regular
            grid, such that `wigner[i, j]` is its value at `(xvec[i], yvec[j])`, as
            returned by [`dq.wigner()`][dynamiqs.wigner].
        xmax: Maximum absolute value of the $x$ coordinate of the grid.
        ymax: Maximum absolute value of the $y$ coordinate of the grid.
        ax: Matplotlib `Axes` to plot on. If `None`, a new figure of size `(w, h)`
            is created.
        w: Width of the new figure in inches, ignored if `ax` is specified.
        h: Height of the new figure in inches, ignored if `ax` is specified. If
            `None`, defaults to `w / 1.6`.
        vmax: Absolute value mapped to the extremal colors of the colormap, values
            outside $[-v_\text{max}, v_\text{max}]$ are clipped. If `None`, defaults
            to $1/(\pi\hbar)$, the maximum absolute value of a Wigner function.
        cmap: Colormap.
        interpolation: Interpolation method passed to `plt.imshow()`.
        colorbar: Whether to add a colorbar.
        cross: Whether to draw the $x=0$ and $y=0$ axes.
        clear: Whether to remove the axes and the colorbar.
        hbar: Value of $\hbar$ used to compute `wigner`, see
            [`dq.wigner()`][dynamiqs.wigner]. It sets the axis labels and the default
            `vmax`.

    Note:
        Choose a diverging colormap `cmap` for better results.

    Examples:
        Plotting pre-computed data makes it possible to post-process it, for example
        to isolate the interference fringes of a cat state by subtracting the Wigner
        function of the corresponding statistical mixture:

        >>> n, alpha = 32, 2.0
        >>> cat = (dq.coherent(n, alpha) + dq.coherent(n, -alpha)).unit()
        >>> mixture = (dq.coherent_dm(n, alpha) + dq.coherent_dm(n, -alpha)) / 2
        >>> _, _, w_cat = dq.wigner(cat, xmax=4.0, ymax=2.0)
        >>> _, _, w_mixture = dq.wigner(mixture, xmax=4.0, ymax=2.0)
        >>> dq.plot.wigner_data(w_cat - w_mixture, 4.0, 2.0)
        >>> renderfig('plot_wigner_data_fringes')

        ![plot_wigner_data_fringes](../../figs_code/plot_wigner_data_fringes.png){.fig-half}

    See also:
        - [`dq.wigner()`][dynamiqs.wigner]: compute the Wigner distribution of a ket or
            density matrix.
        - [`dq.plot.wigner()`][dynamiqs.plot.wigner]: plot the Wigner function of a
            state.
    """
    assert ax is not None

    wig = to_jax(wigner)
    check_shape(wig, 'wigner', '(n, n)')
    if wig.dtype not in (jnp.float32, jnp.float64):
        raise TypeError(
            f'Wigner data must be of type `float`, not `{wig.dtype}`. Consider using'
            f' `dq.plot.wigner(x)` to plot the Wigner function of a quantum state `x`.'
        )

    # set plot norm, |W(x, p)| ≤ 1/(πħ)
    vmax = 1 / (jnp.pi * hbar) if vmax is None else vmax
    vmin = -vmax
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    # clip to avoid rounding errors
    wig = wig.clip(vmin, vmax)

    # plot
    ax.imshow(
        wig.T,
        cmap=cmap,
        norm=norm,
        origin='lower',
        aspect='equal',
        interpolation=interpolation,
        extent=(-xmax, xmax, -ymax, ymax),
    )

    # remove grid by default
    ax.grid(False)

    # axis label, for ħ=0.5 a coherent state |alpha⟩ lies at (Re(alpha), Im(alpha))
    if hbar == 0.5:
        ax.set(xlabel=r'$\mathrm{Re}(\alpha)$', ylabel=r'$\mathrm{Im}(\alpha)$')
    else:
        ax.set(xlabel=r'$x$', ylabel=r'$p$')

    if colorbar and not clear:
        cax = add_colorbar(ax, cmap, norm)
        if vmax == 2 / jnp.pi:
            cax.set_yticks([vmin, 0.0, vmax], labels=[r'$-2/\pi$', r'$0$', r'$2/\pi$'])

    if cross:
        color = '#9e9e9e'
        ax.axhline(0.0, color=color, ls='-', lw=0.7, alpha=0.8)
        ax.axvline(0.0, color=color, ls='-', lw=0.7, alpha=0.8)

    if clear:
        ax.axis(False)


@optional_ax
def wigner(
    state: QArrayLike,
    *,
    ax: Axes | None = None,
    w: float = 7.0,
    h: float | None = None,
    xmax: float = 5.0,
    ymax: float | None = None,
    vmax: float | None = None,
    npixels: int = 101,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    colorbar: bool = True,
    cross: bool = False,
    clear: bool = False,
    hbar: float = 0.5,
):
    r"""Plot the Wigner function of a state.

    Args:
        state (qarray-like of shape (n, 1) or (n, n)): Ket or density matrix.
        ax: Matplotlib `Axes` to plot on. If `None`, a new figure of size `(w, h)`
            is created.
        w: Width of the new figure in inches, ignored if `ax` is specified.
        h: Height of the new figure in inches, ignored if `ax` is specified. If
            `None`, defaults to `w / 1.6`.
        xmax: Maximum absolute value of the $x$ coordinate.
        ymax: Maximum absolute value of the $y$ coordinate. If `None`, defaults to
            `xmax`.
        vmax: Absolute value mapped to the extremal colors of the colormap, values
            outside $[-v_\text{max}, v_\text{max}]$ are clipped. If `None`, defaults
            to $1/(\pi\hbar)$, the maximum absolute value of a Wigner function.
        npixels: Number of pixels in each direction.
        cmap: Colormap.
        interpolation: Interpolation method passed to `plt.imshow()`.
        colorbar: Whether to add a colorbar.
        cross: Whether to draw the $x=0$ and $y=0$ axes.
        clear: Whether to remove the axes and the colorbar.
        hbar: Value of $\hbar$ in the commutation relation $[\hat{x}, \hat{p}]
            = i\hbar$, see [`dq.wigner()`][dynamiqs.wigner].

    Note:
        Choose a diverging colormap `cmap` for better results.

    Warning:
        For the default `hbar=0.5`, the axis scaling is chosen so that a coherent
        state $\ket{\alpha}$ lies at the coordinates
        $(x,y)=(\mathrm{Re}(\alpha),\mathrm{Im}(\alpha))$, which is different from the
        default behaviour of `qutip.plot_wigner()`. For another value of `hbar`, the
        axes are the quadratures $x$ and $p$, and a coherent state lies at
        $(x,p)=\sqrt{2\hbar}\,(\mathrm{Re}(\alpha),\mathrm{Im}(\alpha))$.

    Examples:
        >>> psi = dq.coherent(16, 2.0)
        >>> dq.plot.wigner(psi)
        >>> renderfig('plot_wigner_coh')

        ![plot_wigner_coh](../../figs_code/plot_wigner_coh.png){.fig-half}

        >>> psi = (dq.coherent(16, 2) + dq.coherent(16, -2)).unit()
        >>> dq.plot.wigner(psi, xmax=4.0, ymax=2.0, colorbar=False)
        >>> renderfig('plot_wigner_cat')

        ![plot_wigner_cat](../../figs_code/plot_wigner_cat.png){.fig-half}

        >>> psi = (dq.fock(2, 0) + dq.fock(2, 1)).unit()
        >>> dq.plot.wigner(psi, xmax=2.0, cross=True)
        >>> renderfig('plot_wigner_01')

        ![plot_wigner_01](../../figs_code/plot_wigner_01.png){.fig-half}

        >>> psi = dq.coherent(32, [3, 3j, -3, -3j]).sum(0).unit()
        >>> dq.plot.wigner(psi, npixels=201, clear=True)
        >>> renderfig('plot_wigner_4legged')

        ![plot_wigner_4legged](../../figs_code/plot_wigner_4legged.png){.fig-half}

    See also:
        - [`dq.wigner()`][dynamiqs.wigner]: compute the Wigner distribution of a ket or
            density matrix.
        - [`dq.plot.wigner_data()`][dynamiqs.plot.wigner_data]: plot a pre-computed
            Wigner function.
    """
    assert ax is not None

    state = asqarray(state)
    check_shape(state, 'state', '(n, 1)', '(n, n)')

    ymax = xmax if ymax is None else ymax
    _, _, wig = compute_wigner(state, xmax, ymax, npixels, hbar=hbar)

    wigner_data(
        wig,
        xmax,
        ymax,
        ax=ax,
        vmax=vmax,
        cmap=cmap,
        interpolation=interpolation,
        colorbar=colorbar,
        cross=cross,
        clear=clear,
        hbar=hbar,
    )


def wigner_mosaic(
    states: QArrayLike,
    *,
    n: int = 8,
    nrows: int = 1,
    w: float = 3.0,
    h: float | None = None,
    xmax: float = 5.0,
    ymax: float | None = None,
    vmax: float | None = None,
    npixels: int = 101,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    cross: bool = False,
    hbar: float = 0.5,
):
    r"""Plot the Wigner function of multiple states in a mosaic arrangement.

    Args:
        states (qarray-like of shape (N, n, 1) or (N, n, n)): Sequence of _N_ kets or
            density matrices.
        n: Number of states to plot, evenly spaced in `states` from the first to
            the last state. If `n` exceeds _N_, all states are plotted.
        nrows: Number of rows of the mosaic.
        w: Width of each subplot in inches.
        h: Height of each subplot in inches. If `None`, defaults to `w`.
        xmax: Maximum absolute value of the $x$ coordinate.
        ymax: Maximum absolute value of the $y$ coordinate. If `None`, defaults to
            `xmax`.
        vmax: Absolute value mapped to the extremal colors of the colormap, values
            outside $[-v_\text{max}, v_\text{max}]$ are clipped. If `None`, defaults
            to $1/(\pi\hbar)$, the maximum absolute value of a Wigner function.
        npixels: Number of pixels in each direction.
        cmap: Colormap.
        interpolation: Interpolation method passed to `plt.imshow()`.
        cross: Whether to draw the $x=0$ and $y=0$ axes.
        hbar: Value of $\hbar$ in the commutation relation $[\hat{x}, \hat{p}]
            = i\hbar$, see [`dq.wigner()`][dynamiqs.wigner].

    Examples:
        >>> psis = [dq.fock(3, i) for i in range(3)]
        >>> dq.plot.wigner_mosaic(psis)
        >>> renderfig('plot_wigner_mosaic_fock')

        ![plot_wigner_mosaic_fock](../../figs_code/plot_wigner_mosaic_fock.png){.fig}

        >>> n = 16
        >>> a = dq.destroy(n)
        >>> H = dq.zeros(n)
        >>> jump_ops = [a @ a - 4.0 * dq.eye(n)]  # cat state inflation
        >>> psi0 = dq.coherent(n, 0)
        >>> tsave = jnp.linspace(0, 1.0, 101)
        >>> result = dq.mesolve(H, jump_ops, psi0, tsave)
        >>> dq.plot.wigner_mosaic(result.states, n=6, xmax=4.0, ymax=2.0)
        >>> renderfig('plot_wigner_mosaic_cat')

        ![plot_wigner_mosaic_cat](../../figs_code/plot_wigner_mosaic_cat.png){.fig}

        >>> n = 16
        >>> a = dq.destroy(n)
        >>> H = a.dag() @ a.dag() @ a @ a  # Kerr Hamiltonian
        >>> psi0 = dq.coherent(n, 2)
        >>> tsave = jnp.linspace(0, jnp.pi, 101)
        >>> result = dq.sesolve(H, psi0, tsave)
        >>> dq.plot.wigner_mosaic(result.states, n=25, nrows=5, xmax=4.0)
        >>> renderfig('plot_wigner_mosaic_kerr')

        ![plot_wigner_mosaic_kerr](../../figs_code/plot_wigner_mosaic_kerr.png){.fig}

    See also:
        - [`dq.plot.wigner()`][dynamiqs.plot.wigner]: plot the Wigner function of a
            state.
        - [`dq.plot.wigner_gif()`][dynamiqs.plot.wigner_gif]: plot a GIF of the Wigner
            function of multiple states.
    """
    states = asqarray(states)
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
    selected_indexes = np.linspace(0, nstates - 1, n, dtype=int)
    _, _, wig = compute_wigner(states[selected_indexes], xmax, ymax, npixels, hbar=hbar)

    # plot individual wigner
    for i, ax in enumerate(axs):
        wigner_data(
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
            hbar=hbar,
        )
        ax.set(xlabel='', ylabel='', xticks=[], yticks=[])


def wigner_gif(
    states: QArrayLike,
    *,
    gif_duration: float = 5.0,
    fps: int = 10,
    w: float = 5.0,
    xmax: float = 5.0,
    ymax: float | None = None,
    vmax: float | None = None,
    npixels: int = 101,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    cross: bool = False,
    clear: bool = False,
    hbar: float = 0.5,
) -> Image:
    r"""Plot a GIF of the Wigner function of multiple states.

    If there are more states than GIF frames, the frames show states evenly spaced
    from the first to the last state.

    Args:
        states (qarray-like of shape (N, n, 1) or (N, n, n)): Sequence of _N_ kets or
            density matrices.
        gif_duration: GIF duration in seconds.
        fps: GIF frames per second.
        w: Width of the GIF in inches. The height is set to `w * ymax / xmax`.
        xmax: Maximum absolute value of the $x$ coordinate.
        ymax: Maximum absolute value of the $y$ coordinate. If `None`, defaults to
            `xmax`.
        vmax: Absolute value mapped to the extremal colors of the colormap, values
            outside $[-v_\text{max}, v_\text{max}]$ are clipped. If `None`, defaults
            to $1/(\pi\hbar)$, the maximum absolute value of a Wigner function.
        npixels: Number of pixels in each direction.
        cmap: Colormap.
        interpolation: Interpolation method passed to `plt.imshow()`.
        cross: Whether to draw the $x=0$ and $y=0$ axes.
        clear: Whether to remove the axes.
        hbar: Value of $\hbar$ in the commutation relation $[\hat{x}, \hat{p}]
            = i\hbar$, see [`dq.wigner()`][dynamiqs.wigner].

    Returns:
        (IPython.display.Image): Animated GIF, displayed automatically when it is the
            last expression of a Jupyter notebook cell, see
            [`dq.plot.gifit()`][dynamiqs.plot.gifit] to save it to a file.

    Examples:
        >>> n = 16
        >>> a = dq.destroy(n)
        >>> H = dq.zeros(n)
        >>> jump_ops = [a @ a - 4.0 * dq.eye(n)]  # cat state inflation
        >>> psi0 = dq.coherent(n, 0)
        >>> tsave = jnp.linspace(0, 1.0, 1001)
        >>> result = dq.mesolve(H, jump_ops, psi0, tsave)
        >>> gif = dq.plot.wigner_gif(result.states, fps=25, xmax=4.0, ymax=2.0)
        >>> rendergif(gif, 'wigner-cat')

        ![plot_wigner_gif_cat](../../figs_code/wigner-cat.gif){.fig}

        >>> n = 16
        >>> a = dq.destroy(n)
        >>> H = a.dag() @ a.dag() @ a @ a  # Kerr Hamiltonian
        >>> psi0 = dq.coherent(n, 2)
        >>> tsave = jnp.linspace(0, jnp.pi, 1001)
        >>> result = dq.sesolve(H, psi0, tsave)
        >>> gif = dq.plot.wigner_gif(
        ...     result.states, gif_duration=10.0, fps=25, xmax=4.0, clear=True
        ... )
        >>> rendergif(gif, 'wigner-kerr')

        ![plot_wigner_gif_kerr](../../figs_code/wigner-kerr.gif){.fig-half}

    See also:
        - [`dq.plot.wigner()`][dynamiqs.plot.wigner]: plot the Wigner function of a
            state.
        - [`dq.plot.wigner_mosaic()`][dynamiqs.plot.wigner_mosaic]: plot the Wigner
            function of multiple states in a mosaic arrangement.
        - [`dq.plot.gifit()`][dynamiqs.plot.gifit]: transform a plot function into a
            function that returns an animated GIF.
    """
    states = asqarray(states)
    check_shape(states, 'states', '(N, n, 1)', '(N, n, n)')

    ymax = xmax if ymax is None else ymax
    nframes = int(gif_duration * fps)
    indices = gif_indices(len(states), nframes)
    _, _, wig = compute_wigner(states[indices], xmax, ymax, npixels, hbar=hbar)

    return gifit(wigner_data)(
        list(wig),
        w=w,
        h=ymax / xmax * w,
        xmax=xmax,
        ymax=ymax,
        vmax=vmax,
        cmap=cmap,
        interpolation=interpolation,
        colorbar=False,
        cross=cross,
        clear=clear,
        hbar=hbar,
        gif_duration=gif_duration,
        fps=fps,
    )
