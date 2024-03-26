from __future__ import annotations

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.scipy.linalg import toeplitz
from jaxtyping import ArrayLike

from ..._checks import check_shape
from ..._utils import cdtype
from ..operators import eye
from .general import isket, todm

__all__ = ['wigner']


@partial(jax.jit, static_argnames=('npixels', 'method'))
@partial(jnp.vectorize, signature='(n,m)->(k),(l),(k,l)', excluded={1, 2, 3})
def wigner(
    state: ArrayLike,
    xmax: float = 6.2832,
    ymax: float = 6.2832,
    npixels: int = 200,
    method: Literal['clenshaw', 'fft'] = 'clenshaw',
    g: float = 2.0,
) -> tuple[Array, Array, Array]:
    r"""Compute the Wigner distribution of a ket or density matrix.

    Args:
        state _(array_like of shape (..., n, 1) or (..., n, n))_: Ket or density matrix.
        xmax: Maximum value of x.
        ymax: Maximum value of p. Ignored if the Wigner distribution is computed
            with the `fft` method, in which case `ymax` is given by `2 * pi / xmax`.
        npixels: Number of pixels in each direction.
        method _(string)_: Method used to compute the Wigner distribution. Available
            methods: `'clenshaw'` or `'fft'`.
        g: Scaling factor of Wigner quadratures, such that `a = 0.5 * g * (x + i * p)`.

    Returns:
        A tuple `(xvec, yvec, w)` where

            - xvec: Array of shape _(npixels)_ containing x values
            - yvec: Array of shape _(npixels)_ containing p values
            - w: Array of shape _(npixels, npixels)_ containing the wigner
                distribution at all (x, p) points.
    """
    state = jnp.asarray(state)
    check_shape(state, 'state', '(..., n, 1)', '(..., n, n)')

    xvec = jnp.linspace(-xmax, xmax, npixels)
    yvec = jnp.linspace(-ymax, ymax, npixels)

    if method == 'clenshaw':
        state = todm(state)
        w = _wigner_clenshaw(state, xvec, yvec, g)
    elif method == 'fft':
        if isket(state):
            w, yvec = _wigner_fft_psi(state, xvec, g)
        else:
            w, yvec = _wigner_fft_dm(state, xvec, g)
    else:
        raise ValueError(
            f'Method "{method}" is not supported (supported: "clenshaw", "fft").'
        )

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


def _wigner_fft_psi(psi: Array, xvec: Array, g: float) -> tuple[Array, Array]:
    """Compute the wigner distribution of a ket with the FFT."""
    n = psi.shape[0]

    # compute psi in position basis
    U = _fock_to_position(n, xvec * g / jnp.sqrt(2))
    psi_x = psi.T @ U

    # compute the wigner distribution of psi using the fast Fourier transform
    w, yvec = _wigner_fft(psi_x[0], xvec * g / jnp.sqrt(2))

    return 0.5 * g**2 * w.T.real, yvec * jnp.sqrt(2) / g


def _wigner_fft_dm(rho: Array, xvec: Array, g: float) -> tuple[Array, Array]:
    """Compute the wigner distribution of a density matrix with the FFT."""
    # diagonalize rho
    eig_vals, eig_vecs = lax.linalg.eigh(rho)

    # compute the wigner distribution of each eigenstate
    W = 0
    for i in range(rho.shape[0]):
        eig_vec = eig_vecs[:, i].reshape(rho.shape[0], 1)
        W_psi, yvec = _wigner_fft_psi(eig_vec, xvec, g)
        W += eig_vals[i] * W_psi

    return W, yvec


def _fock_to_position(n: int, positions: Array) -> Array:
    """Compute the change-of-basis matrix from the Fock basis to the position basis of
    an oscillator of dimension n, as evaluated at the specific position values provided.
    """
    n_positions = positions.shape[0]
    U = jnp.zeros((n, n_positions), dtype=cdtype())
    U = U.at[0].set(jnp.pi ** (-0.25) * jnp.exp(-0.5 * positions**2))

    if n == 1:
        return U

    U = U.at[1].set(jnp.sqrt(2.0) * positions * U[0, :])
    for k in range(2, n):
        U = U.at[k].set(
            jnp.sqrt(2.0 / k) * positions * U[k - 1, :]
            - jnp.sqrt(1.0 - 1.0 / k) * U[k - 2, :]
        )
    return U


def _wigner_fft(psi: Array, xvec: Array) -> tuple[Array, Array]:
    """Wigner distribution of a given ket using the fast Fourier transform.

    Args:
        psi: ket of shape (N)
        xvec: position vector of shape (N)

    Returns:
        A tuple `(w, p)` where `w` is the wigner function at all sample points, and `p`
        is the vector of momentum sample points.
    """
    n = len(psi)

    # compute the fourier transform of psi
    r1 = jnp.concatenate(
        (jnp.array([0.0]), jnp.flip(psi.conj(), axis=-1), jnp.zeros(n - 1)), axis=0
    )
    r2 = jnp.concatenate((jnp.array([0.0]), psi, jnp.zeros(n - 1)), axis=0)
    w = toeplitz(jnp.zeros(n), r=r1) * jnp.flipud(toeplitz(jnp.zeros(n), r=r2))
    w = jnp.concatenate((w[:, n : 2 * n], w[:, 0:n]), axis=1)
    w = jnp.fft.fft(w)
    w = jnp.concatenate((w[:, 3 * n // 2 : 2 * n + 1], w[:, 0 : n // 2]), axis=1).real

    # compute the fourier transform of xvec
    p = jnp.arange(-n / 2, n / 2) * jnp.pi / (2 * n * (xvec[1] - xvec[0]))

    # normalize wigner distribution
    w = w / (p[1] - p[0]) / (2 * n)

    return w, p
