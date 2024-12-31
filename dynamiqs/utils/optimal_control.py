from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import ArrayLike

from .._checks import check_shape
from .._utils import cdtype
from ..qarrays.qarray import QArray
from ..qarrays.utils import asqarray
from .operators import displace
from .states import excited, ground

__all__ = ['cd_gate', 'snap_gate']


def snap_gate(phase: ArrayLike) -> QArray:
    r"""Returns a SNAP gate.

    The *selective number-dependent arbitrary phase* (SNAP) gate imparts a different
    phase $\theta_k$ to each Fock state $\ket{k}\bra{k}$. It is defined by
    $$
        \mathrm{SNAP}(\theta_0,\dots,\theta_{n-1}) =
        \sum_{k=0}^{n-1} e^{i\theta_k} \ket{k}\bra{k}.
    $$

    Args:
        phase _(array-like of shape (..., n))_: Phase for each Fock state. The last
            dimension of the array _n_ defines the Hilbert space dimension.

    Returns:
        _(qarray of shape (..., n, n))_ SNAP gate operator.

    Examples:
        >>> dq.snap_gate([0, 1, 2])
        QArray: shape=(3, 3), dims=(3,), dtype=complex64, layout=dense
        [[ 1.   +0.j     0.   +0.j     0.   +0.j   ]
         [ 0.   +0.j     0.54 +0.841j  0.   +0.j   ]
         [ 0.   +0.j     0.   +0.j    -0.416+0.909j]]
        >>> dq.snap_gate([[0, 1, 2], [2, 3, 4]]).shape
        (2, 3, 3)
    """
    phase = jnp.asarray(phase, dtype=cdtype())
    check_shape(phase, 'phase', '(..., n)')

    # batched construct diagonal array
    bdiag = jnp.vectorize(jnp.diag, signature='(a)->(a,a)')
    array = bdiag(jnp.exp(1j * phase))
    return asqarray(array)


def cd_gate(dim: int, alpha: ArrayLike) -> QArray:
    r"""Returns a conditional displacement gate.

    The *conditional displacement* (CD) gate displaces an oscillator conditioned on
    the state of a coupled two-level system (TLS) state. It is defined by
    $$
       \mathrm{CD}(\alpha) = D(\alpha/2)\ket{g}\bra{g} + D(-\alpha/2)\ket{e}\bra{e},
    $$
    where $\ket{g}$ (defined by [`dq.ground()`][dynamiqs.ground]) and $\ket{e}$
    (defined by [`dq.excited()`][dynamiqs.excited]) are the ground and excited states of
    the TLS.

    Args:
        dim: Dimension of the oscillator Hilbert space.
        alpha _(array-like of shape (...))_: Displacement amplitude.

    Returns:
        _(qarray of shape (..., n, n))_ CD gate operator (acting on the oscillator + TLS
            system of dimension _n = 2 x dim_).

    Examples:
        >>> dq.cd_gate(2, 0.1)
        QArray: shape=(4, 4), dims=(2, 2), dtype=complex64, layout=dense
        [[ 0.999+0.j  0.   +0.j  0.05 +0.j  0.   +0.j]
         [ 0.   +0.j  0.999+0.j  0.   +0.j -0.05 +0.j]
         [-0.05 +0.j  0.   +0.j  0.999+0.j  0.   +0.j]
         [ 0.   +0.j  0.05 +0.j  0.   +0.j  0.999+0.j]]
        >>> dq.cd_gate(3, [0.1, 0.2]).shape
        (2, 6, 6)
    """
    alpha = jnp.asarray(alpha, dtype=cdtype())
    g = ground()  # (2, 1)
    e = excited()  # (2, 1)
    disp_plus = displace(dim, alpha / 2)  # (..., dim, dim)
    disp_minus = displace(dim, -alpha / 2)  # (..., dim, dim)
    return (disp_plus & g.proj()) + (disp_minus & e.proj())
