from __future__ import annotations

import jax
from jax import Array
from jaxtyping import PRNGKeyArray

from ..qarrays.qarray import QArray
from ..utils.general import dag, unit

__all__ = ['complex', 'dm', 'herm', 'ket', 'psd', 'real']


def real(
    key: PRNGKeyArray,
    shape: int | tuple[int, ...],
    *,
    min: float = 0.0,  # noqa: A002
    max: float = 1.0,  # noqa: A002
) -> Array:
    r"""Returns an array of uniformly distributed random real numbers.

    Each element of the returned array is sampled uniformly in
    $[\text{min}, \text{max})$.

    Args:
        key: A PRNG key used as the random key.
        shape _(int or tuple of ints)_: Shape of the returned array.
        min: Minimum (inclusive) value.
        max: Maximum (exclusive) value.

    Returns:
        _(array of shape (*shape))_ Random real number array.

    Examples:
        >>> key = jax.random.PRNGKey(42)
        >>> dq.random.real(key, (2, 5), max=5.0)
        Array([[2.444, 3.399, 3.081, 2.805, 2.253],
               [2.929, 0.374, 3.876, 3.495, 4.093]], dtype=float32)
    """
    shape = (shape,) if isinstance(shape, int) else shape
    # sample uniformly in [min, max)
    return jax.random.uniform(key, shape=shape, minval=min, maxval=max)


def complex(  # noqa: A001
    key: PRNGKeyArray, shape: int | tuple[int, ...], *, rmax: float = 1.0
) -> Array:
    r"""Returns an array of uniformly distributed random complex numbers.

    Each element of the returned array is sampled uniformly in the disk of radius
    $\text{rmax}$.

    Note-: Uniform sampling in the complex plane
        Here are three common options to generate random complex numbers,
        `dq.random.complex()` returns the last one:

        ```python
        _, (ax0, ax1, ax2) = dq.plot.grid(3, sharexy=True)
        ax0.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))

        n = 10_000

        # option 1: uniformly distributed real and imaginary part
        x = np.random.rand(n) * 2 - 1 + 1j * (np.random.rand(n) * 2 - 1)
        ax0.scatter(x.real, x.imag, s=1.0)

        # option 2: uniformly distributed magnitude and phase
        x = np.random.rand(n) * jnp.exp(1j * 2 * jnp.pi * np.random.rand(n))
        ax1.scatter(x.real, x.imag, s=1.0)

        # option 3: uniformly distributed in a disk (in Dynamiqs)
        key = jax.random.PRNGKey(42)
        x = dq.random.complex(key, n)
        ax2.scatter(x.real, x.imag, s=1.0)
        renderfig('random_complex')
        ```

        ![rand_complex](../../figs_code/random_complex.png){.fig}

    Args:
        key: A PRNG key used as the random key.
        shape _(int or tuple of ints)_: Shape of the returned array.
        rmax: Maximum magnitude.

    Returns:
        _(array of shape (*shape))_ Random complex number array.

    Examples:
        >>> key = jax.random.PRNGKey(42)
        >>> dq.random.complex(key, (2, 3), rmax=5.0)
        Array([[ 1.617+0.307j,  0.078+2.351j, -0.817+3.812j],
               [-4.221-0.581j, -3.044+2.898j, -0.668+4.552j]], dtype=complex64)
    """
    shape = (shape,) if isinstance(shape, int) else shape
    # sample uniformly in the unit L2 ball and scale
    x = rmax * jax.random.ball(key, 2, shape=shape)
    return x[..., 0] + 1j * x[..., 1]


def herm(key: PRNGKeyArray, shape: tuple[int, ...]) -> QArray:
    """Returns a random complex Hermitian matrix.

    Args:
        key: A PRNG key used as the random key.
        shape _(shape of the form (..., n, n))_: Shape of the returned qarray.

    Returns:
        _(qarray of shape (*shape))_ Random complex Hermitian matrix.

    Examples:
        >>> key = jax.random.PRNGKey(42)
        >>> dq.random.herm(key, (2, 2))
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[ 0.323+0.j    -0.074-0.146j]
         [-0.074+0.146j -0.844+0.j   ]]
    """
    if not len(shape) >= 2 or not shape[-1] == shape[-2]:
        raise ValueError(
            f'Argument `shape` must be of the form (..., n, n), but is shape={shape}.'
        )
    x = complex(key, shape)
    return 0.5 * (x + dag(x))


def psd(key: PRNGKeyArray, shape: tuple[int, ...]) -> QArray:
    """Returns a random complex positive semi-definite matrix.

    Args:
        key: A PRNG key used as the random key.
        shape _(shape of the form (..., n, n))_: Shape of the returned qarray.

    Returns:
        _(qarray of shape (*shape))_ Random complex positive semi-definite matrix.

    Examples:
        >>> key = jax.random.PRNGKey(42)
        >>> dq.random.psd(key, (2, 2))
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[ 0.33 +0.j    -0.074-0.652j]
         [-0.074+0.652j  1.334+0.j   ]]

    """
    if not len(shape) >= 2 or not shape[-1] == shape[-2]:
        raise ValueError(
            f'Argument `shape` must be of the form (..., n, n), but is shape={shape}.'
        )
    x = complex(key, shape)
    return x @ dag(x)


def dm(key: PRNGKeyArray, shape: tuple[int, ...]) -> QArray:
    """Returns a random density matrix (hermitian, positive semi-definite, and unit
    trace).

    Args:
        key: A PRNG key used as the random key.
        shape _(shape of the form (..., n, n))_: Shape of the returned qarray.

    Returns:
        _(qarray of shape (*shape))_ Random density matrix.

    Examples:
        >>> key = jax.random.PRNGKey(42)
        >>> dq.random.dm(key, (2, 2))
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[ 0.198+0.j    -0.044-0.392j]
         [-0.044+0.392j  0.802+0.j   ]]
    """
    if not len(shape) >= 2 or not shape[-1] == shape[-2]:
        raise ValueError(
            f'Argument `shape` must be of the form (..., n, n), but is shape={shape}.'
        )
    x = psd(key, shape)
    return unit(x)


def ket(key: PRNGKeyArray, shape: tuple[int, ...]) -> QArray:
    """Returns a random ket with unit norm.

    Args:
        key: A PRNG key used as the random key.
        shape _(shape of the form (..., n, 1))_: Shape of the returned qarray.

    Returns:
        _(qarray of shape (*shape))_ Random ket.

    Examples:
        >>> key = jax.random.PRNGKey(42)
        >>> dq.random.ket(key, (2, 1))
        QArray: shape=(2, 1), dims=(2,), dtype=complex64, layout=dense
        [[0.563+0.107j]
         [0.027+0.819j]]
    """
    if not len(shape) >= 2 or not shape[-1] == 1:
        raise ValueError(
            f'Argument `shape` must be of the form (..., n, 1), but is shape={shape}.'
        )
    x = complex(key, shape)
    return unit(x)
