from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax
from jaxtyping import ArrayLike

from ..._checks import check_shape
from ..operators import eye
from .general import todm

__all__ = ['wigner']


@partial(jax.jit, static_argnames=('npixels', 'xvec', 'yvec'))
@partial(jnp.vectorize, signature='(n,m)->(k),(l),(k,l)', excluded={1, 2, 3})
def wigner(
    state: ArrayLike,
    xmax: float = 6.2832,
    ymax: float = 6.2832,
    npixels: int = 200,
    xvec: ArrayLike | None = None,
    yvec: ArrayLike | None = None,
    g: float = 2.0,
) -> tuple[Array, Array, Array]:
    r"""Compute the Wigner distribution of a ket or density matrix.

    Args:
        state _(array_like of shape (..., n, 1) or (..., n, n))_: Ket or density matrix.
        xmax: Maximum value of x.
        ymax: Maximum value of p.
        npixels: Number of pixels in each direction.
        xvec: List of x coordinates to compute wigner elements on. If none,
            `xvec = jnp.linspace(-xmax, xmax, npixels)`.
        yvec: List of y coordinates to compute wigner elements on.
            If none, `yvec = jnp.linspace(-ymax, ymax, npixels)`.
        g: Scaling factor of Wigner quadratures, such that `a = 0.5 * g * (x + i * p)`.

    Returns:
        A tuple `(xvec, yvec, w)` where

            - xvec: Array of shape _(npixels)_ containing x values
                or `xvec` if provided in argument
            - yvec: Array of shape _(npixels)_ containing p values
                or `yvec` if provided in argument
            - w: Array of shape _(npixels, npixels)_ containing the wigner
                distribution at all (x, p) points.
    """
    state = jnp.asarray(state)
    check_shape(state, 'state', '(..., n, 1)', '(..., n, n)')

    if xvec is None:
        xvec = jnp.linspace(-xmax, xmax, npixels)
    else:
        check_shape(xvec, 'xvec', '(n,)')

    if yvec is None:
        yvec = jnp.linspace(-ymax, ymax, npixels)
    else:
        check_shape(yvec, 'yvec', '(n,)')

    state = todm(state)
    w = _wigner_clenshaw(state, xvec, yvec, g)

    return xvec, yvec, w


def _wigner_clenshaw(
    rho: ArrayLike, xvec: ArrayLike, yvec: ArrayLike, g: float
) -> Array:
    """Compute the wigner distribution of a density matrix using the iterative method
    of QuTiP based on the Clenshaw summation algorithm.
    """
    rho = jnp.asarray(rho)
    xvec = jnp.asarray(xvec)
    yvec = jnp.asarray(yvec)
    n = rho.shape[-1]

    x, p = jnp.meshgrid(xvec, yvec, indexing='ij')
    a = 0.5 * g * (x + 1.0j * p)
    a2 = jnp.abs(a) ** 2

    w = 2 * rho[0, -1] * jnp.ones_like(a)
    rho = rho * (2 * jnp.ones((n, n)) - eye(n))

    def loop(i: int, w: Array) -> Array:
        i = n - 2 - i
        w = w * (2 * a * (i + 1) ** (-0.5))
        return w + (_laguerre_series(i, 4 * a2, rho, n))

    w = lax.fori_loop(0, n - 1, loop, w)

    return (w.real * jnp.exp(-2 * a2) * 0.5 * g**2 / jnp.pi).T


def _diag_element(mat: jnp.array, diag: int, element: int) -> float:
    r"""Return the element of a matrix `mat` at `jnp.diag(mat, diag)[element]`.
    This function is jittable for `diag` while it is not for the `jnp.diag` version.
    """
    assert mat.shape[0] == mat.shape[1], 'Matrix must be square.'
    n = mat.shape[0]
    element = jax.lax.select(element < 0, n - jnp.abs(diag) - jnp.abs(element), element)
    return mat[jnp.maximum(-diag, 0) + element, jnp.maximum(diag, 0) + element]


def _laguerre_series(i: int, x: Array, rho: Array, n: int) -> Array:
    r"""Evaluate a polynomial series of the form `$\sum_n c_n L_n^i$` where `$L_n$` is
    such that `$L_n^i = (-1)^n \sqrt(i!n!/(i+n)!) LaguerreL[n,i,x]$`.
    """

    def n_1() -> Array:
        return _diag_element(rho, i, 0) * jnp.ones_like(x)

    def n_2() -> Array:
        c0 = _diag_element(rho, i, 0)
        c1 = _diag_element(rho, i, 1)
        return (c0 - c1 * (i + 1 - x) * (i + 1) ** (-0.5)) * jnp.ones_like(x)

    def n_other() -> Array:
        cm2 = _diag_element(rho, i, -2)
        cm1 = _diag_element(rho, i, -1)
        y0 = cm2 * jnp.ones_like(x)
        y1 = cm1 * jnp.ones_like(x)

        def loop(j: int, args: tuple[Array, Array]) -> tuple[Array, Array]:
            k = n + 1 - i - j
            y0, y1 = args
            ckm1 = _diag_element(rho, i, -j)
            y0, y1 = (
                ckm1 - y1 * (k * (i + k) / ((i + k + 1) * (k + 1))) ** 0.5,
                y0 - y1 * (i + 2 * k - x + 1) * ((i + k + 1) * (k + 1)) ** -0.5,
            )

            return y0, y1

        y0, y1 = lax.fori_loop(3, n + 1 - i, loop, (y0, y1))

        return y0 - y1 * (i + 1 - x) * (i + 1) ** (-0.5)

    return lax.cond(n - i == 1, n_1, lambda: lax.cond(n - i == 2, n_2, n_other))
