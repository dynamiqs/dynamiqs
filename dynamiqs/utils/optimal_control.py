from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from ..utils.operators import displace
from ..utils.states import fock
from ..utils.utils import tensor, tobra
from .array_types import cdtype

__all__ = ['rand_real', 'rand_complex', 'snap_gate', 'cd_gate']


def rand_real(
    key: PRNGKeyArray,
    shape: int | tuple[int, ...],
    *,
    min: float = 0.0,
    max: float = 1.0,
) -> Array:
    r"""Returns an array filled with uniformly distributed random real numbers.

    Each element of the returned array is sampled uniformly in
    $[\text{min}, \text{max})$.

    Args:
        key: A PRNG key used as the random key.
        shape _(int or tuple of ints)_: Shape of the returned array.
        min: Minimum (inclusive) value.
        max: Maximum (exclusive) value.

    Returns:
        _(array of shape (*shape))_ Array filled with random real numbers.

    Examples:
        >>> key = jax.random.PRNGKey(42)
        >>> dq.rand_real(key, (2, 5), max=5.0)
        Array([[3.22 , 1.613, 0.967, 4.432, 4.21 ],
               [0.96 , 1.726, 1.262, 3.16 , 3.274]], dtype=float32)
    """
    shape = (shape,) if isinstance(shape, int) else shape
    # sample uniformly in [min, max)
    return jax.random.uniform(key, shape=shape, minval=min, maxval=max)


def rand_complex(
    key: PRNGKeyArray, shape: int | tuple[int, ...], *, rmax: float = 1.0
) -> Array:
    r"""Returns an array filled with random complex numbers uniformly distributed in
    the complex plane.

    Each element of the returned array is sampled uniformly in the disk of radius
    $\text{rmax}$.

    Notes-: Uniform sampling in the complex plane
        Here are three common options to generate random complex numbers,
        `dq.rand_complex()` returns the last one:

        ```python
        _, (ax0, ax1, ax2) = dq.gridplot(3, sharex=True, sharey=True)
        ax0.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))

        n = 10_000

        # option 1: uniformly distributed real and imaginary part
        x = np.random.rand(n) * 2 - 1 + 1j * (np.random.rand(n) * 2 - 1)
        ax0.scatter(x.real, x.imag, s=1.0)

        # option 2: uniformly distributed magnitude and phase
        x = np.random.rand(n) * np.exp(1j * 2 * np.pi * np.random.rand(n))
        ax1.scatter(x.real, x.imag, s=1.0)

        # option 3: uniformly distributed in a disk (in dynamiqs)
        key = jax.random.PRNGKey(42)
        x = dq.rand_complex(key, n)
        ax2.scatter(x.real, x.imag, s=1.0)
        renderfig('rand_complex')
        ```

        ![rand_complex](/figs-code/rand_complex.png){.fig}

    Args:
        key: A PRNG key used as the random key.
        shape _(int or tuple of ints)_: Shape of the returned array.
        rmax: Maximum magnitude.

    Returns:
        _(array of shape (*shape))_ Array filled with random complex numbers.

    Examples:
        >>> key = jax.random.PRNGKey(42)
        >>> dq.rand_complex(key, (2, 3), rmax=5.0)
        Array([[ 1.341+4.17j ,  3.978-0.979j, -2.592-0.946j],
               [-4.428+1.744j, -0.53 +1.668j,  2.582+0.65j ]], dtype=complex64)
    """
    shape = (shape,) if isinstance(shape, int) else shape
    # sample uniformly in the unit L2 ball and scale
    x = rmax * jax.random.ball(key, 2, shape=shape)
    return x[..., 0] + 1j * x[..., 1]


def snap_gate(phase: ArrayLike) -> Array:
    r"""Returns a SNAP gate.

    The *selective number-dependent arbitrary phase* (SNAP) gate imparts a different
    phase $\theta_k$ to each Fock state $\ket{k}\bra{k}$. It is defined by
    $$
        \mathrm{SNAP}(\theta_0,\dots,\theta_{n-1}) =
        \sum_{k=0}^{n-1} e^{i\theta_k} \ket{k}\bra{k}.
    $$

    Args:
        phase _(array_like of shape (..., n))_: Phase for each Fock state. The last
            dimension of the array _n_ defines the Hilbert space dimension.

    Returns:
        _(array of shape (..., n, n))_ SNAP gate operator.

    Examples:
        >>> dq.snap_gate([0, 1, 2])
        Array([[ 1.   +0.j   ,  0.   +0.j   ,  0.   +0.j   ],
               [ 0.   +0.j   ,  0.54 +0.841j,  0.   +0.j   ],
               [ 0.   +0.j   ,  0.   +0.j   , -0.416+0.909j]], dtype=complex64)
        >>> dq.snap_gate([[0, 1, 2], [2, 3, 4]]).shape
        (2, 3, 3)
    """
    phase = jnp.asarray(phase)
    # batched construct diagonal array
    bdiag = jnp.vectorize(jnp.diag, signature='(a)->(a,a)')
    return bdiag(jnp.exp(1j * phase))


def cd_gate(dim: int, alpha: ArrayLike) -> Array:
    r"""Returns a conditional displacement gate.

    The *conditional displacement* (CD) gate displaces an oscillator conditioned on
    the state of a coupled two-level system (TLS) state. It is defined by
    $$
       \mathrm{CD}(\alpha) = D(\alpha/2)\ket{g}\bra{g} + D(-\alpha/2)\ket{e}\bra{e},
    $$
    where $\ket{g}=\ket0$ and $\ket{e}=\ket1$ are the ground and excited states of the
    TLS, respectively.

    Args:
        dim: Dimension of the oscillator Hilbert space.
        alpha _(array_like of shape (...))_: Displacement amplitude.

    Returns:
        _(array of shape (..., n, n))_ CD gate operator (acting on the oscillator + TLS
            system of dimension _n = 2 x dim_).

    Examples:
        >>> dq.cd_gate(2, 0.1)
        Array([[ 0.999+0.j,  0.   +0.j, -0.05 +0.j,  0.   +0.j],
               [ 0.   +0.j,  0.999+0.j,  0.   +0.j,  0.05 +0.j],
               [ 0.05 +0.j,  0.   +0.j,  0.999+0.j,  0.   +0.j],
               [ 0.   +0.j, -0.05 +0.j,  0.   +0.j,  0.999+0.j]], dtype=complex64)
        >>> dq.cd_gate(3, [0.1, 0.2]).shape
        (2, 6, 6)
    """  # noqa: E501
    alpha = jnp.asarray(alpha, dtype=cdtype())
    g = fock(2, 0)  # (2, 1)
    e = fock(2, 1)  # (2, 1)
    disp_plus = displace(dim, alpha / 2)  # (..., dim, dim)
    disp_minus = displace(dim, -alpha / 2)  # (..., dim, dim)
    return tensor(disp_plus, g @ tobra(g)) + tensor(disp_minus, e @ tobra(e))
