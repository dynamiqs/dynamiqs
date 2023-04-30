from __future__ import annotations

from math import pi, sqrt
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from .tensor_types import dtype_float_to_complex
from .utils import is_ket, ket_to_dm

__all__ = ['plot_wigner', 'wigner']


def plot_wigner(
    state: Tensor,
    *,
    x_max: float = 6.2832,
    p_max: float = 6.2832,
    n_pixels: int = 200,
    x_lim: float = 6.0,
    p_lim: float = 6.0,
    method: Literal['clenshaw', 'fft'] = 'clenshaw',
) -> mpl.figure.Figure:
    """Plot the wigner distribution of a state vector of density matrix.

    Args:
        state: State vector or density matrix.
        x_max: Maximum value of x for which to compute the wigner distribution.
        p_max: Maximum value of p for which to compute the wigner distribution.
            If the wigner distribution is computed using the `fft` method, `p_max` is
            ignored, and given by `2 * pi / x_max` instead.
        n_pixels: Number of pixels in each direction.
        x_lim: Limit of x for the displayed plot.
        p_lim: Limit of p for the displayed plot.
        method: Method used to compute the wigner distribution.

    Returns:
        The figure object corresponding to the displayed plot.
    """
    # compute wigner function
    xvec, pvec, W = wigner(
        state, x_max=x_max, p_max=p_max, n_pixels=n_pixels, method=method
    )
    xvec, pvec, W = xvec.detach().numpy(), pvec.detach().numpy(), W.detach().numpy()

    # make figure
    fig = plt.figure()
    ax = plt.axes(xlim=(-x_lim, x_lim), ylim=(-p_lim, p_lim))
    ax.set_aspect('equal', adjustable='box')

    # plot
    cmap = mpl.colormaps['RdBu']
    norm = mpl.colors.Normalize(-abs(W).max(), abs(W).max())
    ax.contourf(xvec, pvec, W, 100, norm=norm, cmap=cmap)
    ax.set_xlabel('x')
    ax.set_ylabel('p')

    # display
    plt.tight_layout()
    plt.show()

    return fig


def wigner(
    state: Tensor,
    x_max: float = 6.2832,
    p_max: float = 6.2832,
    n_pixels: int = 200,
    method: Literal['clenshaw', 'fft'] = 'clenshaw',
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the wigner distribution of a state vector or density matrix.

    Args:
        state: State vector or density matrix.
        x_max: Maximum value of x for which to compute the wigner distribution.
        p_max: Maximum value of p for which to compute the wigner distribution.
            If the wigner distribution is computed using the `fft` method, `p_max` is
            ignored, and given by `2 * pi / x_max` instead.
        n_pixels: Number of pixels in each direction.
        method: Method used to compute the wigner distribution.

    Returns:
        A tuple `(xvec, pvec, w)` where
            xvec: 1D Tensor of x values
            pvec: 1D Tensor of p values
            w: 2D Tensor with the wigner distribution
    """
    if state.ndim > 2:
        raise NotImplementedError('Batching is not yet implemented for `wigner`.')

    xvec = torch.linspace(-x_max, x_max, n_pixels)
    pvec = torch.linspace(-p_max, p_max, n_pixels)

    if method == 'clenshaw':
        state = ket_to_dm(state) if is_ket(state) else state
        w = _wigner_clenshaw(state, xvec, pvec)
    elif method == 'fft':
        if is_ket(state):
            w, pvec = _wigner_fft_psi(state, xvec)
        else:
            w, pvec = _wigner_fft_dm(state, xvec)
    else:
        raise ValueError(f'Method {method} does not exist.')

    return xvec, pvec, w


def _wigner_clenshaw(rho: Tensor, xvec: Tensor, pvec: Tensor):
    """Compute the wigner distribution of a density matrix using the iterative method
    of QuTiP based on the Clenshaw summation algorithm."""
    n = rho.size(-1)

    x, p = torch.meshgrid(xvec, pvec)
    a = sqrt(2) * (x + 1.0j * p)
    a2 = a.abs() ** 2

    w = 2 * rho[0, n - 1] * torch.ones_like(a)
    rho = rho * (2 * torch.ones(n, n) - torch.diag(torch.ones(n)))
    i = n - 1
    for i in range(n - 2, -1, -1):
        w *= a * (i + 1) ** (-0.5)
        w += _wigner_laguerre_series(i, a2, torch.diag(rho, i))

    return w.real * torch.exp(-0.5 * a2) / pi


def _wigner_laguerre_series(i, x, c):
    r"""Evaluate a polynomial series of the form `$\sum_n c_n L_n^i$` where `$L_n$` is
    such that `$L_n^i = (-1)^n \sqrt(i!n!/(i+n)!) LaguerreL[n,i,x]$`."""
    n = len(c)
    if n == 1:
        y0 = c[0]
        y1 = 0
    elif n == 2:
        y0 = c[0]
        y1 = c[1]
    else:
        y0 = c[-2]
        y1 = c[-1]
        for k in range(n - 2, 0, -1):
            y0, y1 = (
                c[k - 1] - y1 * (k * (i + k) / ((i + k + 1) * (k + 1))) ** 0.5,
                y0 - y1 * (i + 2 * k - x + 1) * ((i + k + 1) * (k + 1)) ** -0.5,
            )

    return y0 - y1 * (i + 1 - x) * (i + 1) ** (-0.5)


def _wigner_fft_psi(psi: Tensor, xvec: Tensor) -> tuple[Tensor, Tensor]:
    """Compute the wigner distribution of a state vector with the FFT."""
    n = psi.size(0)

    # compute psi in position basis
    U = _fock_to_position(n, xvec)
    psi_x = psi.T @ U

    # compute the wigner distribution of psi using the fast Fourier transform
    w, pvec = _wigner_fft(psi_x[0], xvec)

    return w.T.real, pvec


def _wigner_fft_dm(rho: Tensor, xvec: Tensor) -> tuple[Tensor, Tensor]:
    """Compute the wigner distribution of a density matrix with the FFT."""
    # diagonalize rho
    eig_vals, eig_vecs = torch.linalg.eigh(rho)

    # compute the wigner distribution of each eigenstate
    W = 0
    for i in range(rho.shape[0]):
        eig_vec = eig_vecs[:, i].reshape(rho.shape[0], 1)
        W_psi, pvec = _wigner_fft_psi(eig_vec, xvec)
        W += eig_vals[i] * W_psi

    return W, pvec


def _fock_to_position(n: int, positions: Tensor) -> Tensor:
    """
    Compute the change-of-basis matrix from the Fock basis to the position basis of an
    oscillator of dimension n, as evaluated at the specific position values provided.
    """
    n_positions = positions.shape[0]
    U = torch.zeros(n, n_positions, dtype=dtype_float_to_complex(positions.dtype))
    U[0, :] = pi ** (-0.25) * torch.exp(-0.5 * positions**2)

    if n == 1:
        return U

    U[1, :] = sqrt(2) * positions * U[0, :]
    for k in range(2, n):
        U[k, :] = sqrt(2.0 / k) * positions * U[k - 1, :]
        U[k, :] -= sqrt(1.0 - 1.0 / k) * U[k - 2, :]
    return U


def _wigner_fft(psi: Tensor, xvec: Tensor) -> tuple[Tensor, Tensor]:
    """Wigner distribution of a given state vector using the fast Fourier transform.

    Args:
        psi: state vector of size (N)
        xvec: position vector of size (N)
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
