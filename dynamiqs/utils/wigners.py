from __future__ import annotations

from math import pi, sqrt
from typing import Literal

import torch
from torch import Tensor

from .tensor_types import dtype_real_to_complex
from .utils import isdm, isket, todm

__all__ = ['wigner']


def wigner(
    state: Tensor,
    x_max: float = 6.2832,
    p_max: float = 6.2832,
    num_pixels: int = 200,
    method: Literal['clenshaw', 'fft'] = 'clenshaw',
    g: float = 2.0,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Compute the Wigner distribution of a ket or density matrix.

    Args:
        state _(..., n, 1) or (..., n, n)_: Ket or density matrix.
        x_max: Maximum value of x.
        p_max: Maximum value of p. Ignored if the Wigner distribution is computed
            with the `fft` method, in which case `p_max` is given by `2 * pi / x_max`.
        num_pixels: Number of pixels in each direction.
        method _(string)_: Method used to compute the Wigner distribution. Available
            methods: `'clenshaw'` or `'fft'`.
        g: Scaling factor of Wigner quadratures, such that `a = 0.5 * g * (x + i * p)`.

    Returns:
        A tuple `(xvec, pvec, w)` where

            - xvec: Tensor of shape _(num_pixels)_ containing x values
            - pvec: Tensor of shape _(num_pixels)_ containing p values
            - w: Tensor of shape _(num_pixels, num_pixels)_ containing the wigner
                distribution at all (x, p) points.
    """
    if state.ndim > 2:
        raise NotImplementedError('Batching is not yet implemented for `wigner`.')

    xvec = torch.linspace(-x_max, x_max, num_pixels + 1)
    pvec = torch.linspace(-p_max, p_max, num_pixels + 1)

    if method == 'clenshaw':
        state = todm(state)
        w = _wigner_clenshaw(state, xvec, pvec, g)
    elif method == 'fft':
        if isket(state):
            w, pvec = _wigner_fft_psi(state, xvec, g)
        elif isdm(state):
            w, pvec = _wigner_fft_dm(state, xvec, g)
        else:
            raise ValueError(
                'Input state must be a ket or density matrix, but got shape'
                f' {state.shape}.'
            )
    else:
        raise ValueError(
            f'Method "{method}" is not supported (supported: "clenshaw", "fft").'
        )

    return xvec, pvec, w


def _wigner_clenshaw(rho: Tensor, xvec: Tensor, pvec: Tensor, g: float):
    """Compute the wigner distribution of a density matrix using the iterative method
    of QuTiP based on the Clenshaw summation algorithm."""
    n = rho.size(-1)

    x, p = torch.meshgrid(xvec, pvec, indexing='ij')
    a = 0.5 * g * (x + 1.0j * p)
    a2 = a.abs() ** 2

    w = 2 * rho[0, -1] * torch.ones_like(a)
    rho = rho * (2 * torch.ones(n, n) - torch.diag(torch.ones(n)))
    for i in range(n - 2, -1, -1):
        w *= 2 * a * (i + 1) ** (-0.5)
        w += _laguerre_series(i, 4 * a2, torch.diag(rho, i))

    return (w.real * torch.exp(-2 * a2) * 0.5 * g**2 / pi).T


def _laguerre_series(i, x, c):
    r"""Evaluate a polynomial series of the form `$\sum_n c_n L_n^i$` where `$L_n$` is
    such that `$L_n^i = (-1)^n \sqrt(i!n!/(i+n)!) LaguerreL[n,i,x]$`."""
    n = len(c)

    if n == 1:
        return c[0]
    elif n == 2:
        return c[0] - c[1] * (i + 1 - x) * (i + 1) ** (-0.5)

    y0 = c[-2]
    y1 = c[-1]
    for k in range(n - 2, 0, -1):
        y0, y1 = (
            c[k - 1] - y1 * (k * (i + k) / ((i + k + 1) * (k + 1))) ** 0.5,
            y0 - y1 * (i + 2 * k - x + 1) * ((i + k + 1) * (k + 1)) ** -0.5,
        )

    return y0 - y1 * (i + 1 - x) * (i + 1) ** (-0.5)


def _wigner_fft_psi(psi: Tensor, xvec: Tensor, g: float) -> tuple[Tensor, Tensor]:
    """Compute the wigner distribution of a ket with the FFT."""
    n = psi.size(0)

    # compute psi in position basis
    U = _fock_to_position(n, xvec * g / sqrt(2))
    psi_x = psi.T @ U

    # compute the wigner distribution of psi using the fast Fourier transform
    w, pvec = _wigner_fft(psi_x[0], xvec * g / sqrt(2))

    return 0.5 * g**2 * w.T.real, pvec * sqrt(2) / g


def _wigner_fft_dm(rho: Tensor, xvec: Tensor, g: float) -> tuple[Tensor, Tensor]:
    """Compute the wigner distribution of a density matrix with the FFT."""
    # diagonalize rho
    eig_vals, eig_vecs = torch.linalg.eigh(rho)

    # compute the wigner distribution of each eigenstate
    W = 0
    for i in range(rho.shape[0]):
        eig_vec = eig_vecs[:, i].reshape(rho.shape[0], 1)
        W_psi, pvec = _wigner_fft_psi(eig_vec, xvec, g)
        W += eig_vals[i] * W_psi

    return W, pvec


def _fock_to_position(n: int, positions: Tensor) -> Tensor:
    """
    Compute the change-of-basis matrix from the Fock basis to the position basis of an
    oscillator of dimension n, as evaluated at the specific position values provided.
    """
    n_positions = positions.shape[0]
    U = torch.zeros(n, n_positions, dtype=dtype_real_to_complex(positions.dtype))
    U[0, :] = pi ** (-0.25) * torch.exp(-0.5 * positions**2)

    if n == 1:
        return U

    U[1, :] = sqrt(2) * positions * U[0, :]
    for k in range(2, n):
        U[k, :] = sqrt(2.0 / k) * positions * U[k - 1, :]
        U[k, :] -= sqrt(1.0 - 1.0 / k) * U[k - 2, :]
    return U


def _wigner_fft(psi: Tensor, xvec: Tensor) -> tuple[Tensor, Tensor]:
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
    r1 = torch.cat((torch.tensor([0]), psi.conj().flip(-1), torch.zeros(n - 1)))
    r2 = torch.cat((torch.tensor([0]), psi, torch.zeros(n - 1)))
    w = _toeplitz(torch.zeros(n), r1) * _toeplitz(torch.zeros(n), r2).flipud()
    w = torch.cat((w[:, n : 2 * n], w[:, 0:n]), dim=1)
    w = torch.fft.fft(w)
    w = torch.cat((w[:, 3 * n // 2 : 2 * n + 1], w[:, 0 : n // 2]), axis=1).real

    # compute the fourier transform of xvec
    p = torch.arange(-n / 2, n / 2) * pi / (2 * n * (xvec[1] - xvec[0]))

    # normalize wigner distribution
    w = w / (p[1] - p[0]) / (2 * n)

    return w, p


def _toeplitz(c: Tensor, r: Tensor = None) -> Tensor:
    """Construct a Toeplitz matrix.

    Code copied from https://stackoverflow.com/a/68899386/9099342.
    """
    c = torch.ravel(c)
    r = torch.ravel(r) if r is not None else c.conj()
    vals = torch.cat((r, c[1:].flip(0)))
    shape = len(c), len(r)
    i, j = torch.ones(*shape).nonzero().T
    return vals[j - i].reshape(*shape)
