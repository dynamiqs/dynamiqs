from __future__ import annotations

from cmath import exp
from math import sinh, sqrt

import torch
from numpy import angle
from scipy.special import factorial, factorial2, iv
from torch import Tensor

from ..utils.operators import parity, quadrature
from ..utils.tensor_types import get_cdtype

__all__ = ['quadrature_sign', 'twophdiss_invariants']


def quadrature_sign(
    dim: int,
    phi: float,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the quadrature sign operator of phase angle $\phi$.

    It is defined by $s(\phi) = \mathrm{sign}(e^{i\phi} a^\dag + e^{-i\phi} a)$, where
    $a$ and $a^\dag$ are the annihilation and creation operators respectively.

    Args:
        dim: Dimension of the Hilbert space.
        phi: Phase angle.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(dim, dim)_ Quadrature sign operator.
    """
    quad = quadrature(dim, phi, dtype=get_cdtype(dtype), device=device)
    L, Q = torch.linalg.eigh(quad)
    sign_L = torch.diag(torch.sign(L)).to(Q.dtype)
    return Q @ sign_L @ Q.mH


def twophdiss_invariants(
    dim: int,
    alpha: float,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns the Jx, Jy, and Jz two-photon dissipation invariants.

    These are defined in Mirrahimi, Mazyar, et al. "Dynamically protected cat-qubits: a
    new paradigm for universal quantum computation." New Journal of Physics 16.4 (2014).
    They correspond to the invariants of the dynamnics described by the two-photon
    dissipation superoperator $D[a^2 - \alpha^2]$.

    Args:
        dim: Dimension of the Hilbert space.
        alpha: Complex two-photon squeezing amplitude.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        - _(dim, dim)_: Jx invariant
        - _(dim, dim)_: Jy invariant
        - _(dim, dim)_: Jz invariant
    """
    # complex dtype
    cdtype = get_cdtype(dtype)

    # Jx invariant
    Jx = parity(dim, dtype=cdtype, device=device)

    # Jy, Jz invariant
    Jpm = torch.zeros((dim, dim), dtype=cdtype, device=device)
    for q in range(-dim // 2, dim // 2):
        if q >= 0:
            n = torch.arange(0, dim - (2 * q + 1), 2)
            n_1 = torch.where(n == 0, 0, n - 1)
            values = (
                _torch_factorial2(n_1)
                / _torch_factorial2(n + 2 * q)
                * torch.sqrt(_torch_factorial(n + 2 * q + 1) / _torch_factorial(n))
            )
            diag = torch.zeros(dim - (2 * q + 1), dtype=cdtype, device=device)
            diag[::2] = values
            Jpmq = torch.diag(diag, diagonal=2 * q + 1)
        if q < 0:
            n = torch.arange(0, dim + 2 * q, 2)
            values = (
                _torch_factorial2(n + 1)
                / _torch_factorial2(n - 2 * q)
                * torch.sqrt(_torch_factorial(n - 2 * q) / _torch_factorial(n + 1))
            )
            diag = torch.zeros(dim + 2 * q + 1, dtype=cdtype, device=device)
            diag[1::2] = values
            Jpmq = torch.diag(diag, diagonal=-2 * q - 1)
        Jpm += (
            (-1) ** abs(q)
            / (2 * q + 1)
            * iv(q, abs(alpha) ** 2)
            * exp(-1j * (2 * q + 1) * angle(alpha))
            * Jpmq
        )

    Jpm *= sqrt((2 * abs(alpha) ** 2) / (sinh(2 * abs(alpha) ** 2)))
    Jy = 1j * Jpm - 1j * Jpm.conj().T
    Jz = Jpm + Jpm.conj().T

    return Jx, Jy, Jz


def _torch_factorial(x: Tensor) -> Tensor:
    """Returns the factorial of n."""
    return torch.from_numpy(factorial(x)).to(x)


def _torch_factorial2(x: Tensor) -> Tensor:
    """Returns the double factorial of n."""
    return torch.from_numpy(factorial2(x)).to(x)
