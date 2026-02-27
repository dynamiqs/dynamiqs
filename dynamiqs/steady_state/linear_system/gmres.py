import functools as ft
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

MatVec = Callable[[Array], Array]
BoolLike = Array
BoolFn = Callable[[Array], BoolLike]

Carry = tuple[Array, Array, Array, Array, Array]
ArnoldiCarry = tuple[Array, Array, Array, Array, Array, Array]


def gmres(
    A: MatVec,
    M: MatVec,
    x_0: Array,
    b: Array,
    stopping_criterion: BoolFn,
    max_iter: int,
    krylov_size: int,
    recycling: int | tuple[Array, Array],
) -> tuple[Array, tuple[Array, Array, Array, Array]]:
    """GMRES with _right_ preconditioning, recycling, and custom stopping criterion.

    Args:
        A: linear operator
        M: right preconditioner
        x_0: initial guess
        b: right-hand side
        stopping_criterion: function that returns True when converged
        max_iter: maximum number of GMRES cycles
        krylov_size: size of Krylov subspace per cycle
        recycling: either int (number of vectors to recycle, 0 for no recycling)
                   or tuple (U, C) of existing recycling arrays

    Returns:
        x: solution
        info: tuple (n_iterations, success, U, C) where U, C are recycling data
    """
    assert_dtypes_are_the_same(A, M, x_0, b)
    dtype = x_0.dtype
    (dimension,) = x_0.shape

    U, C = initialize_recycling_arrays(recycling, dimension, dtype)
    V, H = initialize_krylov_arrays(dimension, krylov_size, dtype)

    # First GMRES cycle to initialize recycled space
    x_first, U_first, C_first, V_first, H_bar_first, converged_at_first = (
        gmres_one_cycle(A, M, x_0, b, U, C, V, H, stopping_criterion)
    )
    # Initialize recycling arrays if recycling > 0 and U, C were empty
    U, C = maybe_update_recycling_arrays(
        U_first, C_first, V_first, H_bar_first, recycling
    )

    def cond_fn(carry: Carry) -> BoolLike:
        n_iteration, _x, _U, _C, converged = carry
        return (n_iteration < max_iter - 1) & (jnp.logical_not(converged))

    def body_fn(carry: Carry) -> Carry:
        n_iteration, x, U, C, converged = carry
        x_new, U_new, C_new, _, _, converged = gmres_one_cycle(
            A, M, x, b, U, C, V, H, stopping_criterion
        )
        return (n_iteration + 1, x_new, U_new, C_new, converged)

    n_iteration, x, U, C, success = jax.lax.while_loop(
        cond_fn, body_fn, (1, x_first, U, C, converged_at_first)
    )
    return x, (n_iteration, success, U, C)


def gmres_one_cycle(
    A: MatVec,
    M: MatVec,
    x_0: Array,
    b: Array,
    U: Array,
    C: Array,
    V: Array,
    H: Array,
    stopping_criterion: BoolFn,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """One cycle of GMRES with right-preconditioning and recycling.

    Args:
        A: linear operator
        M: right preconditioner
        x_0: current solution guess
        b: right-hand side
        U: recycled solution basis (dimension, p), satisfies (AM) U = C
        C: recycled image basis (dimension, p), orthonormal columns
        V: storage for Krylov basis (dimension, krylov_size)
        H: storage for Hessenberg matrix (krylov_size, krylov_size - 1)

    Returns:
        solution: updated solution x_new
        U_new: updated recycled solution basis
        C_new: updated recycled image basis
        V_out: Krylov basis from this cycle (for recycling initialization)
        H_bar: augmented Hessenberg matrix (for recycling initialization)

    Notes:
        Will implement Givens rotation later.

    Algorithm:
        1. Compute residual r_0 = b - A x_0
        2. Deflation: alpha = C^H r_0 (projection onto recycled space)
        3. Arnoldi with recycling: builds V, H, B from deflated residual
        4. Build augmented Hessenberg: H_bar = [[I_p, B], [0, H]]
        5. Solve min ||[alpha, beta, 0..0]^T - H_bar y|| via QR
        6. Correction: x_new = x_0 + M(U y_U + V_{n-1} y_V)
        7. Update recycling via harmonic Ritz extraction
    """
    r_0 = b - A(x_0)
    p = C.shape[1]

    # Compute deflation coefficients for RHS
    # alpha = C^H r_0 gives the projection of r_0 onto the recycled space C
    alpha = C.conj().mT @ r_0  # (p,)

    # Arnoldi with deflation and recycling
    # arnoldi internally computes r_proj = r_0 - C alpha and builds Krylov space
    V, H, B, beta, breakdown = arnoldi(lambda x: A(M(x)), U, C, V, H, r_0)

    # Build augmented Hessenberg matrix H_bar = [[I_p, B], [0, H]]
    H_bar = build_augmented_hessenberg_matrix(B, H)

    # Build RHS for least squares: [alpha, beta, 0..0]^T
    # - alpha: deflation coefficients (p entries)
    # - beta: norm of deflated residual
    # - zeros: remaining entries (n-1 entries)
    n = H.shape[0]  # krylov_size
    rhs = jnp.concatenate([alpha, jnp.zeros(n, dtype=alpha.dtype).at[0].set(beta)])

    # Solve least squares min ||rhs - H_bar y|| via QR decomposition
    Q_h, R_h = jnp.linalg.qr(H_bar, mode='reduced')
    y = jnp.linalg.solve(R_h, Q_h.conj().T @ rhs)

    # Split y into coefficients for recycled space and Krylov space
    y_U = y[:p]  # (p,) coefficients for U
    y_V = y[p:]  # (n-1,) coefficients for V_{n-1}

    # Compute correction in preconditioned space and apply preconditioner
    # correction = M(U y_U + V_{n-1} y_V)
    correction = M(U @ y_U + V[:, :-1] @ y_V)
    solution = x_0 + correction

    converged = stopping_criterion(solution)
    V, H, B = _error_if_breakdown((V, H, B), breakdown & jnp.logical_not(converged))

    # Update recycling arrays via harmonic Ritz extraction
    # Keep the same number of recycled vectors (p_new = p)
    # For first cycle with p=0, this returns empty arrays;
    # initialization happens in maybe_update_recycling_arrays
    U_new, C_new = extract_recycled_matrices(H_bar, U, V, C, p)

    return solution, U_new, C_new, V, H_bar, converged


def arnoldi(
    A: MatVec, U: Array, C: Array, V: Array, H: Array, r_0: Array
) -> tuple[Array, Array, Array, Array, BoolLike]:
    r"""Builds the Krylov subspace associated to `(A, r_0)` with recycling from `(U, C)`
    .

    Performs initial deflation against C, then builds the Krylov space orthogonal to C.

    Specifically, the recycled space is
    ```
        U, s.t. A U = C
        C orthogonal
    ```
    and it builds the Krylov space
    ```
        Span(r_proj, A r_proj, ..., A^(n-1) r_proj) = Span V_n
    ```
    where `r_proj = r_0 - C (C^H r_0)` is the deflated residual.

    The relation is:
    ```
        A [U V_(n-1)] = [C V] H_bar
    ```
    where H_bar = [[I_p, B], [0, H]] (built separately in gmres_one_cycle).

    Args:
        A: matrix-vector product operator
        U: recycled solution basis (dimension, p)
        C: recycled image basis (dimension, p), orthonormal columns
        V: storage for Krylov basis (dimension, krylov_size)
        H: storage for Hessenberg matrix (krylov_size, krylov_size - 1)
        r_0: initial residual (dimension,)

    Returns:
        V: orthonormal Krylov basis (dimension, krylov_size)
        H: Hessenberg matrix (krylov_size, krylov_size - 1)
        B: recycling coefficients (p, krylov_size - 1)
        beta: norm of deflated residual
        breakdown: whether the Arnoldi process broke down

    Note:
        Our `H` (size `(n, n-1)`) is usually denoted as `\\tilde{H}` in textbooks.

        We use a `scan` loop rather than a while loop, deferring the breakdown check
        at the very end.
    """
    m, n = V.shape
    _, p = C.shape
    assert U.shape == C.shape
    assert U.shape[0] == m
    assert H.shape == (n, n - 1)
    assert r_0.shape == (m,)

    # Step 1: Initial deflation against C
    # Project r_0 onto C and subtract to get orthogonal component
    # When C is empty (p=0), this is a no-op
    r_proj = r_0 - C @ (C.conj().mT @ r_0)

    # Reset storage variables -- will happen in-place with Jax optimizations
    V = jnp.zeros_like(V)
    H = jnp.zeros_like(H)
    B = jnp.zeros((p, n - 1), dtype=V.dtype)

    beta = jnp.linalg.norm(r_proj)
    (_, H, B, V, q_end, h_end), _ = jax.lax.scan(
        ft.partial(_arnoldi_inner, A=A, C=C),
        (0, H, B, V, r_proj / beta, 0.0),
        length=n - 1,
    )
    # Fills the last values
    V = V.at[:, -1].set(q_end)
    H = H.at[-1, -1].set(h_end)

    # Breakdown is only tested at the end of the computation of the whole
    # Krylov space.
    breakdown = jnp.isclose(h_end, 0.0) | jnp.isnan(h_end)

    return V, H, B, beta, breakdown


def _arnoldi_inner(
    carry: ArnoldiCarry, _: object, *, A: MatVec, C: Array
) -> tuple[ArnoldiCarry, None]:
    """Inner loop of Arnoldi iteration with recycling.

    Called by the scan function. Second argument is input of scan, which is unused.

    Args:
        carry: (k, H, B, V, q_new, h_res) where
            k: iteration index
            H: Hessenberg matrix (krylov_size, krylov_size - 1)
            B: recycling coefficients matrix (p, krylov_size - 1)
            V: orthonormal basis (dimension, krylov_size)
            q_new: new candidate vector for V
            h_res: residual norm from previous step
        A: matrix-vector product operator
        C: recycled image basis (dimension, p), orthonormal columns
    """
    k, H, B, V, q_new, h_res = carry

    # Update V and H
    # V is simply writing the new vector
    V = V.at[:, k].set(q_new)
    # For H, we begin by filling the subdiagonal element of the PREVIOUS round
    # At iteration k=0, we are setting the value in [0, -1] to h_res = 0.
    # Not the most stylish, but it avoids an additional condition check on the
    # first pass.
    H = H.at[k, k - 1].set(h_res)

    # Classical Gram-Schmidt process
    w = A(q_new)

    # Orthogonalize against C (recycled space)
    # When C is empty (p=0), this is a no-op due to JAX empty array optimization
    b = C.conj().mT @ w  # (p,)
    w = w - C @ b
    B = B.at[:, k].set(b)

    # Orthogonalize against V
    # In what follows, we could replace:
    # - V with V[:, :k]
    # - H with H.at[:k, k].set(h)
    # unfortunately, k is a traced array so this would require dynamic indexing
    # todo: test performance with dynamic indexing
    h = V.conj().mT @ w  # (krylov_size,)
    w = w - V @ h

    # Re-orthogonalization against C and V to improve stability
    # This is crucial for recycled GMRES because the augmented Hessenberg
    # system assumes exact orthogonality between [C, V_n].
    b2 = C.conj().mT @ w
    w = w - C @ b2
    B = B.at[:, k].add(b2)

    h2 = V.conj().mT @ w
    w = w - V @ h2
    h = h + h2

    # We do not perform a breakdown check of the Arnoldi iteration
    # If the vectors becomes too colinear, we will get nan values
    # todo: handle breakdown by checking h_res
    # and using a random vector orthogonal to V if necessary
    h_res = jnp.linalg.norm(w)
    q_new = w / h_res

    H = H.at[:, k].set(h)

    return (k + 1, H, B, V, q_new, h_res), None


def _error_if_breakdown(
    args: tuple[Array, Array, Array], breakdown: BoolLike
) -> tuple[Array, Array, Array]:
    return eqx.error_if(
        args,
        breakdown,
        'Arnoldi process resulted in colinear basis. Not Handled for now.',
    )


def build_augmented_hessenberg_matrix(B: Array, H_aug: Array) -> Array:
    """Build the augmented Hessenberg matrix H_bar for recycled GMRES.

    The combined relation is:
    ```
        A [U, V_{n-1}] = [C, V_n] H_bar
    ```

    where H_bar has the block structure:
    ```
        H_bar = [[I_p, B   ],
                 [0,   H_aug]]
    ```

    Args:
        B: recycling coefficients (p, n-1) from orthogonalization against C
        H_aug: Hessenberg matrix (n, n-1) from Arnoldi process

    Returns:
        H_bar: augmented Hessenberg matrix (p + n, p + n - 1)

    Note:
        When p=0 (no recycling), B is empty and H_bar = H_aug.
    """
    p, n_minus_1 = B.shape
    n, _ = H_aug.shape
    assert H_aug.shape == (n, n_minus_1)

    # Build H_bar = [[I_p, B], [0, H_aug]]
    # Top-left: I_p (p x p)
    # Top-right: B (p x n-1)
    # Bottom-left: 0 (n x p)
    # Bottom-right: H_aug (n x n-1)

    top = jnp.concatenate([jnp.eye(p, dtype=B.dtype), B], axis=1)  # (p, p + n - 1)
    bottom = jnp.concatenate(
        [jnp.zeros((n, p), dtype=H_aug.dtype), H_aug], axis=1
    )  # (n, p + n - 1)

    return jnp.concatenate([top, bottom], axis=0)  # (p + n, p + n - 1) (H_bar)


def extract_recycled_matrices(
    H_bar: Array, U: Array, V: Array, C: Array, p_new: int
) -> tuple[Array, Array]:
    """Extract new recycled subspaces via harmonic Ritz vectors.

    Computes the best `p_new` vectors to keep for the next GMRES cycle by
    extracting the harmonic Ritz vectors corresponding to the smallest
    singular values of `H_bar`.

    Args:
        H_bar: augmented Hessenberg matrix (p_old + n, p_old + n - 1)
        U: current recycled solution basis (dimension, p_old)
        V: Krylov basis from Arnoldi (dimension, n), includes V_{n+1}
        C: current recycled image basis (dimension, p_old)
        p_new: number of vectors to extract for recycling

    Returns:
        U_new: new recycled solution basis (dimension, p_new)
        C_new: new recycled image basis (dimension, p_new), orthonormal

    Note:
        When p_new=0, returns empty arrays.
        For first cycle initialization: p_old=0, p_new=recycling.
        For subsequent cycles: p_old=p_new=p (keep same number).
    """
    # When p_new=0, return empty arrays
    dimension = V.shape[0]
    dtype = V.dtype

    if p_new == 0:
        return (
            jnp.zeros((dimension, 0), dtype=dtype),
            jnp.zeros((dimension, 0), dtype=dtype),
        )

    # 1. SVD of H_bar to get harmonic Ritz vectors
    # H_bar = U_sigma @ Sigma @ Vh
    # We want the last p_new columns of V (right singular vectors)
    # corresponding to smallest singular values
    _, _, Vh = jnp.linalg.svd(H_bar, full_matrices=True)
    # Vh has shape (p_old + n - 1, p_old + n - 1), rows are right singular vectors
    # G_p is the last p_new columns of V = Vh.conj().T
    # Note: we use H_bar.shape[1] - p_new to avoid the -0 indexing issue
    n_cols = H_bar.shape[1]
    G_p = Vh.conj().T[:, n_cols - p_new :]  # (p_old + n - 1, p_new)

    # 2. Form candidate subspaces
    # [U, V_{n-1}] are the columns we're combining
    V_n_minus_1 = V[:, :-1]  # (dimension, n-1)
    UV = jnp.concatenate([U, V_n_minus_1], axis=1)  # (dimension, p_old + n - 1)

    # [C, V_n] for the image space
    CV = jnp.concatenate([C, V], axis=1)  # (dimension, p_old + n)

    U_tilde = UV @ G_p  # (dimension, p_new)
    C_tilde = CV @ (H_bar @ G_p)  # (dimension, p_new)

    # 3. QR orthonormalization to enforce C_new^H C_new = I
    C_new, R = jnp.linalg.qr(C_tilde, mode='reduced')  # C_new: (dimension, p_new)

    # 4. U_new = U_tilde @ R^{-1} to maintain A U_new = C_new
    # Solve R @ U_new.T = U_tilde.T for U_new.T
    U_new = jnp.linalg.solve(R, U_tilde.T).T  # (dimension, p_new)

    return U_new, C_new


def assert_dtypes_are_the_same(A: MatVec, M: MatVec, x_0: Array, b: Array) -> None:
    dtype_output_of_A = jax.eval_shape(lambda: A(x_0)).dtype
    dtype_output_of_M = jax.eval_shape(lambda: M(x_0)).dtype

    if not dtype_output_of_A == dtype_output_of_M == x_0.dtype == b.dtype:
        raise ValueError(
            f'Dtypes are not compatible. Got '
            f'A output: {dtype_output_of_A}, '
            f'M output: {dtype_output_of_M}, '
            f'x_0: {x_0.dtype}, '
            f'b: {b.dtype}'
        )


def initialize_recycling_arrays(
    recycling: int | tuple[Array, Array], dimension: int, dtype: jnp.dtype
) -> tuple[Array, Array]:
    """Returns the parameters for recycling.

    If recycling is an integer, it means that no recycling has been done so far.
    So we initialize an empty array which will be optimized away under JAX.
    It will be updated later in `maybe_update_recycling_arrays`.
    """
    if isinstance(recycling, int):
        U, C = (
            jnp.zeros((dimension, 0), dtype=dtype),
            jnp.zeros((dimension, 0), dtype=dtype),
        )
    elif isinstance(recycling, tuple):
        U, C = recycling
        if U.shape != C.shape:
            raise ValueError(
                'Recycling arrays must have the same '
                f'shape, got U: {U.shape}, C: {C.shape}'
            )
    else:
        raise TypeError(
            f'Invalid recycling parameter. Expected int, or tuple[Array, Array], '
            f'got {type(recycling)}'
        )
    return U, C


def maybe_update_recycling_arrays(
    U: Array, C: Array, V: Array, H_bar: Array, recycling: int | tuple[Array, Array]
) -> tuple[Array, Array]:
    """Initialize or preserve recycling arrays after first GMRES cycle.

    This function is called after the first GMRES cycle to potentially
    initialize the recycled subspace when recycling > 0 but U, C were empty.

    Args:
        U: current recycled solution basis (from gmres_one_cycle)
        C: current recycled image basis (from gmres_one_cycle)
        V: Krylov basis from first cycle
        H_bar: augmented Hessenberg from first cycle
        recycling: either int (number of vectors to recycle) or existing (U, C)

    Returns:
        U_new: recycled solution basis (possibly initialized)
        C_new: recycled image basis (possibly initialized)
    """
    if recycling == 0:
        # No recycling requested, keep empty arrays
        return U, C
    elif isinstance(recycling, int):
        # recycling > 0: initialize recycled subspace from first cycle
        # U, C are currently empty, extract `recycling` vectors from Arnoldi data
        U_new, C_new = extract_recycled_matrices(H_bar, U, V, C, recycling)
        return U_new, C_new
    elif isinstance(recycling, tuple):
        # Existing recycling data was passed, use the updated arrays
        return U, C
    else:
        raise ValueError(
            f'Invalid recycling parameter. Expected int, or tuple[Array, Array], '
            f'got {type(recycling)}'
        )


def initialize_krylov_arrays(
    dimension: int, krylov_size: int, dtype: jnp.dtype
) -> tuple[Array, Array]:
    return (
        jnp.zeros((dimension, krylov_size), dtype),
        jnp.zeros((krylov_size, krylov_size - 1), dtype),
    )
