from __future__ import annotations

import jax.numpy as jnp

from ..qarrays.qarray import QArray
from ..qarrays.utils import asqarray
from ..utils.operators import quadrature
from ..utils.quantum_utils import dag

__all__ = ['quadrature_sign']


def quadrature_sign(dim: int, phi: float) -> QArray:
    r"""Returns the quadrature sign operator of phase angle $\phi$.

    It is defined by $s_\phi = \mathrm{sign}(e^{i\phi} a^\dag + e^{-i\phi} a)$, where
    $a$ and $a^\dag$ are the annihilation and creation operators respectively.

    Args:
        dim: Dimension of the Hilbert space.
        phi: Phase angle.

    Returns:
        _(qarray of shape (dim, dim))_ Quadrature sign operator.
    """
    quad = quadrature(dim, phi)
    L, Q = quad._eigh()  # noqa: SLF001
    sign_L = jnp.diag(jnp.sign(L))
    array = Q @ sign_L @ dag(Q)
    return asqarray(array)
