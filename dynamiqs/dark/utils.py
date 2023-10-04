from __future__ import annotations

import torch
from torch import Tensor

from ..utils.operators import quadrature

__all__ = ['quadrature_sign']


def quadrature_sign(
    dim: int,
    phi: float,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the quadrature sign operator of phase angle $\phi$.

    It is defined by $s(\phi) = \mathrm{sign}(e^{i\phi} a + e^{-i\phi} a^\dag)$, where
    $a$ and $a^\dag$ are the annihilation and creation operators respectively.

    Args:
        dim: Dimension of the Hilbert space.
        phi: Phase angle.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(dim, dim)_ Quadrature sign operator.
    """
    quad = quadrature(dim, phi, dtype=dtype, device=device)
    L, Q = torch.linalg.eigh(quad)
    sign_L = torch.diag(torch.sign(L)).to(Q.dtype)
    return Q @ sign_L @ Q.mH
