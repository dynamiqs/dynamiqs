from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike
from matplotlib.axes import Axes

from .._checks import check_times
from .utils import colors, optional_ax

__all__ = ['pwc_pulse']


@optional_ax
def pwc_pulse(
    times: ArrayLike,
    values: ArrayLike,
    *,
    ax: Axes | None = None,
    w: float = 7.0,
    h: float | None = None,
    ycenter: bool = True,
    real_color: str = colors['blue'],
    imag_color: str = colors['purple'],
):
    """Plot a piecewise constant pulse.

    The real and imaginary parts of the pulse are plotted separately.

    Args:
        times (array-like of shape (N+1,)): Time points between which the pulse takes
            constant values, defining _N_ time intervals.
        values (array-like of shape (N,)): Pulse value on each time interval.
        ax: Matplotlib `Axes` to plot on. If `None`, a new figure of size `(w, h)`
            is created.
        w: Width of the new figure in inches, ignored if `ax` is specified.
        h: Height of the new figure in inches, ignored if `ax` is specified. If
            `None`, defaults to `w / 1.6`.
        ycenter: Whether to center the y axis on zero.
        real_color: Color of the real part.
        imag_color: Color of the imaginary part.

    Examples:
        >>> n = 20
        >>> times = jnp.linspace(0, 1.0, n + 1)
        >>> key = jax.random.PRNGKey(42)
        >>> values = dq.random.complex(key, n)
        >>> dq.plot.pwc_pulse(times, values)
        >>> renderfig('plot_pwc_pulse')

        ![plot_pwc_pulse](../../figs_code/plot_pwc_pulse.png){.fig}
    """
    assert ax is not None

    times = jnp.asarray(times)  # (n + 1)
    values = jnp.asarray(values)  # (n)
    times = check_times(times, 'times')

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
        ymin, ymax = ax.get_ylim()
        ymax_abs = max(abs(ymin), abs(ymax))
        ax.set_ylim(ymin=-ymax_abs, ymax=ymax_abs)

    ax.set(xlim=(0, times[-1]))
