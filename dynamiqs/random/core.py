from __future__ import annotations

from math import prod

import jax
from jax import Array
from jaxtyping import PRNGKeyArray

from ..qarrays.qarray import QArray
from ..qarrays.utils import asqarray
from ..utils.general import dag

__all__ = ['complex', 'dm', 'herm', 'ket', 'operator', 'psd', 'real']


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
        shape (int or tuple of ints): Shape of the returned array.
        min: Minimum (inclusive) value.
        max: Maximum (exclusive) value.

    Returns:
        (array of shape (*shape)): Random real number array.

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
        shape (int or tuple of ints): Shape of the returned array.
        rmax: Maximum magnitude.

    Returns:
        (array of shape (*shape)): Random complex number array.

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
        shape (shape of the form (..., n, n)): Shape of the returned qarray.

    Returns:
        (qarray of shape (*shape)): Random complex Hermitian matrix.

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
        shape (shape of the form (..., n, n)): Shape of the returned qarray.

    Returns:
        (qarray of shape (*shape)): Random complex positive semi-definite matrix.

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


def dm(
    key: PRNGKeyArray, dims: int | tuple[int, ...], *, batch: int | tuple[int, ...] = ()
) -> QArray:
    """Returns a random density matrix (hermitian, positive semi-definite, and unit
    trace).

    Args:
        key: A PRNG key used as the random key.
        dims: Hilbert space dimension of each subsystem.
        batch: Batch shape of the returned qarray.

    Returns:
        (qarray of shape (*batch, prod(dims), prod(dims))): Random density matrix.

    Examples:
        >>> key = jax.random.PRNGKey(42)

        Random density matrix for an individual system:
        >>> dq.random.dm(key, 2)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[ 0.198+0.j    -0.044-0.392j]
         [-0.044+0.392j  0.802+0.j   ]]

        Batched random density matrices for an individual system:
        >>> dq.random.dm(key, 2, batch=3)
        QArray: shape=(3, 2, 2), dims=(2,), dtype=complex64, layout=dense
        [[[ 0.198+0.j    -0.044-0.392j]
          [-0.044+0.392j  0.802+0.j   ]]
        <BLANKLINE>
         [[ 0.476+0.j     0.021+0.06j ]
          [ 0.021-0.06j   0.524+0.j   ]]
        <BLANKLINE>
         [[ 0.473+0.j    -0.286-0.173j]
          [-0.286+0.173j  0.527+0.j   ]]]

        Random density matrix for a composite system:
        >>> dq.random.dm(key, (2, 3))
        QArray: shape=(6, 6), dims=(6,), dtype=complex64, layout=dense
        [[ 0.178+0.j     0.012+0.021j -0.047+0.057j -0.007+0.072j -0.092-0.033j
           0.033+0.002j]
         [ 0.012-0.021j  0.16 +0.j    -0.019-0.036j  0.013-0.089j -0.003+0.031j
          -0.118-0.052j]
         [-0.047-0.057j -0.019+0.036j  0.137+0.j     0.024+0.012j  0.015+0.03j
           0.041-0.007j]
         [-0.007-0.072j  0.013+0.089j  0.024-0.012j  0.176+0.j     0.007+0.056j
          -0.026-0.087j]
         [-0.092+0.033j -0.003-0.031j  0.015-0.03j   0.007-0.056j  0.133+0.j
          -0.05 -0.002j]
         [ 0.033-0.002j -0.118+0.052j  0.041+0.007j -0.026+0.087j -0.05 +0.002j
           0.216+0.j   ]]

        Batched random density matrices for a composite system:
        >>> dq.random.dm(key, (2, 3), batch=(4, 5, 6, 7)).shape
        (4, 5, 6, 7, 6, 6)
    """
    dims = (dims,) if isinstance(dims, int) else dims
    batch = (batch,) if isinstance(batch, int) else batch
    shape = (*batch, prod(dims), prod(dims))
    x = asqarray(psd(key, shape), dims=dims)
    return x.unit()


def operator(
    key: PRNGKeyArray,
    dims: int | tuple[int, ...],
    hermitian: bool = True,
    *,
    batch: int | tuple[int, ...] = (),
) -> QArray:
    """Returns a random operator.

    Args:
        key: A PRNG key used as the random key.
        dims: Hilbert space dimension of each subsystem.
        hermitian: Whether to return a Hermitian operator.
        batch: Batch shape of the returned qarray.

    Returns:
        (qarray of shape (*batch, prod(dims), prod(dims))): Random operator.

    Examples:
        >>> key = jax.random.PRNGKey(42)

        Random Hermitian operator for an individual system:
        >>> dq.random.operator(key, 2)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[ 0.323+0.j    -0.074-0.146j]
         [-0.074+0.146j -0.844+0.j   ]]

        Random non-Hermitian operator for an individual system:
        >>> dq.random.operator(key, 2, hermitian=False)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[ 0.323+0.061j  0.016+0.47j ]
         [-0.163+0.762j -0.844-0.116j]]

        Batched random Hermitian operators for an individual system:
        >>> dq.random.operator(key, 2, batch=3)
        QArray: shape=(3, 2, 2), dims=(2,), dtype=complex64, layout=dense
        [[[ 0.323+0.j    -0.074-0.146j]
          [-0.074+0.146j -0.844+0.j   ]]
        <BLANKLINE>
         [[-0.609+0.j    -0.082+0.881j]
          [-0.082-0.881j  0.703+0.j   ]]
        <BLANKLINE>
         [[-0.06 +0.j    -0.12 +0.433j]
          [-0.12 -0.433j -0.273+0.j   ]]]

        Random Hermitian operator for a composite system:
        >>> dq.random.operator(key, (2, 3))
        QArray: shape=(6, 6), dims=(6,), dtype=complex64, layout=dense
        [[ 0.323+0.j    -0.008+0.661j -0.254+0.501j  0.004+0.093j -0.581+0.114j
          -0.359+0.114j]
         [-0.008-0.661j  0.703+0.j    -0.381+0.178j -0.167-0.305j  0.245-0.401j
          -0.597-0.076j]
         [-0.254-0.501j -0.381-0.178j  0.469+0.j     0.143+0.173j  0.457+0.518j
          -0.331-0.218j]
         [ 0.004-0.093j -0.167+0.305j  0.143-0.173j  0.685+0.j     0.613+0.355j
           0.201+0.151j]
         [-0.581-0.114j  0.245+0.401j  0.457-0.518j  0.613-0.355j -0.215+0.j
           0.269-0.797j]
         [-0.359-0.114j -0.597+0.076j -0.331+0.218j  0.201-0.151j  0.269+0.797j
           0.655+0.j   ]]

        Random batched Hermitian operators for a composite system:
        >>> dq.random.operator(key, (2, 3), batch=(4, 5, 6, 7)).shape
        (4, 5, 6, 7, 6, 6)
    """
    dims = (dims,) if isinstance(dims, int) else dims
    batch = (batch,) if isinstance(batch, int) else batch
    shape = (*batch, prod(dims), prod(dims))
    x = herm(key, shape) if hermitian else complex(key, shape)
    return asqarray(x, dims=dims)


def ket(
    key: PRNGKeyArray, dims: int | tuple[int, ...], *, batch: int | tuple[int, ...] = ()
) -> QArray:
    r"""Returns a random ket with unit norm.

    Args:
        key: A PRNG key used as the random key.
        dims: Hilbert space dimension of each subsystem.
        batch: Batch shape of the returned qarray.

    Returns:
        (qarray of shape (*batch, prod(dims), 1)): Random ket.

    Examples:
        >>> key = jax.random.PRNGKey(42)

        Random ket for an individual system:
        >>> dq.random.ket(key, 2)
        QArray: shape=(2, 1), dims=(2,), dtype=complex64, layout=dense
        [[0.563+0.107j]
         [0.027+0.819j]]

        Batched random kets for an individual system:
        >>> dq.random.ket(key, 2, batch=3)
        QArray: shape=(3, 2, 1), dims=(2,), dtype=complex64, layout=dense
        [[[ 0.563+0.107j]
          [ 0.027+0.819j]]
        <BLANKLINE>
         [[-0.141+0.66j ]
          [-0.731-0.101j]]
        <BLANKLINE>
         [[-0.489+0.465j]
          [-0.107+0.73j ]]]

        Random ket for a composite system:
        >>> dq.random.ket(key, (2, 3))
        QArray: shape=(6, 1), dims=(2, 3), dtype=complex64, layout=dense
        [[ 0.18 +0.034j]
         [ 0.009+0.262j]
         [-0.091+0.425j]
         [-0.471-0.065j]
         [-0.339+0.323j]
         [-0.074+0.508j]]

        Batched random kets for a composite system:
        >>> dq.random.ket(key, (2, 3), batch=(4, 5, 6, 7)).shape
        (4, 5, 6, 7, 6, 1)
    """
    dims = (dims,) if isinstance(dims, int) else dims
    batch = (batch,) if isinstance(batch, int) else batch
    shape = (*batch, prod(dims), 1)
    x = asqarray(complex(key, shape), dims=dims)
    return x.unit()
