from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import factorial

from ..utils.operators import parity, quadrature
from ..utils.utils import dag

__all__ = ['quadrature_sign', 'twophdiss_invariants']

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


def twophdiss_invariants(
    dim: int,
    alpha: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns the Jx, Jy, and Jz two-photon dissipation invariants.

    These are defined in Mirrahimi, Mazyar, et al. "Dynamically protected cat-qubits: a
    new paradigm for universal quantum computation." New Journal of Physics 16.4 (2014).
    They correspond to the invariants of the dynamnics described by the two-photon
    dissipation superoperator $\mathcal{D}[a^2 - \alpha^2]$.

    Args:
        dim: Dimension of the Hilbert space.
        alpha: Complex two-photon squeezing amplitude.

    Returns:
        - _(dim, dim)_: Jx invariant
        - _(dim, dim)_: Jy invariant
        - _(dim, dim)_: Jz invariant
    """
    # Jx invariant
    Jx = parity(dim, dtype=cdtype, device=device)

    # Jpm invariant
    Jpm = jnp.zeros((dim, dim), dtype=cdtype, device=device)
    for q in range(-dim // 2, dim // 2):
        if q >= 0:
            n = jnp.arange(0, dim - (2 * q + 1), 2)
            n_1 = jnp.where(n == 0, 0, n - 1)
            values = (
                _torch_factorial2(n_1)
                / _torch_factorial2(n + 2 * q)
                * jnp.sqrt(factorial(n + 2 * q + 1) / factorial(n))
            )
            diag = jnp.zeros(dim - (2 * q + 1), dtype=cdtype, device=device)
            diag[::2] = values
            Jpmq = jnp.diag(diag, diagonal=2 * q + 1)
        if q < 0:
            n = jnp.arange(0, dim + 2 * q, 2)
            values = (
                _torch_factorial2(n + 1)
                / _torch_factorial2(n - 2 * q)
                * jnp.sqrt(factorial(n - 2 * q) / factorial(n + 1))
            )
            diag = jnp.zeros(dim + 2 * q + 1, dtype=cdtype, device=device)
            diag[1::2] = values
            Jpmq = jnp.diag(diag, diagonal=-2 * q - 1)
        Jpm += (
            (-1) ** abs(q)
            / (2 * q + 1)
            * iv(q, abs(alpha) ** 2)
            * jnp.exp(-1j * (2 * q + 1) * angle(alpha))
            * Jpmq
        )

    Jpm *= jnp.sqrt((2 * abs(alpha) ** 2) / (jnp.sinh(2 * abs(alpha) ** 2)))

    # Jy, Jz invariants
    Jy = 1j * Jpm - 1j * Jpm.conj().T
    Jz = Jpm + Jpm.conj().T

    return Jx, Jy, Jz
