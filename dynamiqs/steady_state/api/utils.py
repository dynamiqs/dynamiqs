import abc
from collections.abc import Callable
import jax.numpy as jnp
from jax import Array


def finalize_density_matrix(dm: Array, exact_dm: bool) -> Array:
    """Projects the density matrix `rho` onto the set of density matrices with the following properties:
    - Hermitian
    - Positive semi-definite
    - Trace equal to 1
    """
    dm = (dm + dm.conj().mT) / 2
    if exact_dm:
        rank = dm.shape[0]
        dm = projection_dm(dm, rank)
    else:
        dm /= jnp.trace(dm)
    return dm


def update_preconditioner(
    precond_fn: Callable[[Array], Array],
    identity_vectorized: Array,
    use_rank_1_update: bool,
    coefficient: float = 1,
) -> Callable[[Array], Array]:
    """Updates a preconditioner `S` with the Sherman-Moorison formula.
    ```
            inv(S + u u.T)(x) = inv(S)(x) - (u.T inv(S)(x)) / (1 + u.T inv(S)(x)) inv(S)(u)
    ```
    """
    if not use_rank_1_update:
        return precond_fn
    precond_on_identity = precond_fn(identity_vectorized)
    trace_precond_on_identity = identity_vectorized.dot(precond_on_identity)

    def precond_with_rank1(X: Array) -> Array:
        precond_on_X = precond_fn(X)
        trace_precond_on_X = identity_vectorized.dot(precond_on_X)
        return (
            precond_on_X
            - coefficient
            / (1 + trace_precond_on_identity)
            * trace_precond_on_X
            * precond_on_identity
        )

    return precond_with_rank1


def frobenius_dot_product(A, B):
    """Frobenius dot-product."""
    return (A.conj() * B).sum((-1, -2))


def projection_dm(rho: Array, rank: int) -> Array:
    """Projects the density matrix `rho` onto the set of density matrices with rank `rank`."""
    w, v = jnp.linalg.eigh(rho)
    assert isinstance(w, Array)
    w = jnp.where(w < 0, 0.0, w)
    w_new = jnp.zeros_like(w).at[-rank:].set(proj_simplex(w[-rank:]))
    return (v * w_new) @ v.conj().T


def proj_simplex(x: Array) -> Array:
    """Projects the vector `x` onto the simplex."""
    u = jnp.sort(x, descending=True)
    projection = (1 - jnp.cumsum(u)) / (jnp.arange(1, len(x) + 1))
    index = jnp.searchsorted(x + projection <= 0, True)
    theta = projection[index]
    return jnp.maximum(x + theta, 0.0)
