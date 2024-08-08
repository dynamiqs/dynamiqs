from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ..utils.operators import quadrature
from ..utils.quantum_utils import dag

__all__ = ['quadrature_sign']


def quadrature_sign(dim: int, phi: float) -> Array:
    r"""Returns the quadrature sign operator of phase angle $\phi$.

    It is defined by $s_\phi = \mathrm{sign}(e^{i\phi} a^\dag + e^{-i\phi} a)$, where
    $a$ and $a^\dag$ are the annihilation and creation operators respectively.

    Args:
        dim: Dimension of the Hilbert space.
        phi: Phase angle.

    Returns:
        _(array of shape (dim, dim))_ Quadrature sign operator.
    """
    quad = quadrature(dim, phi)
    L, Q = jnp.linalg.eigh(quad)
    sign_L = jnp.diag(jnp.sign(L))
    return Q @ sign_L @ dag(Q)
