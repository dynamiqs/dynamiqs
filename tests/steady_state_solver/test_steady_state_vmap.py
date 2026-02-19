"""Tests for batched (vmap) execution of the steadystate solver.

Uses simple driven-damped oscillators with varying drive strengths and decay
rates so we can build batched QArrays and verify that the batched result
matches element-wise sequential solves.
"""

import jax
import jax.numpy as jnp
import dynamiqs as dq
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 12  # Hilbert space size (kept small for speed)
TOL = 1e-3
KRYLOV = 32
MAX_ITER = 100


def _build_driven_damped_oscillator(n, epsilon, kappa):
    """Build H = epsilon * (a + a†), L = [sqrt(kappa) * a]."""
    a = dq.destroy(n)
    H = epsilon * (a + a.dag())
    L = jnp.sqrt(kappa) * a
    return H, [L]


def _solve_single(H, Ls):
    """Solve a single (non-batched) system and return the result."""
    return dq.steadystate(H, Ls, tol=TOL, max_iteration=MAX_ITER, krylov_size=KRYLOV)


def _assert_rhos_close(rho_batched, rho_ref, idx, atol=1e-3):
    """Assert that a single slice of a batched rho matches a reference rho."""
    rho_b = rho_batched[idx].to_jax()
    rho_r = rho_ref.to_jax()
    assert jnp.allclose(rho_b, rho_r, atol=atol), (
        f'Batched rho[{idx}] does not match sequential result '
        f'(max diff = {float(jnp.max(jnp.abs(rho_b - rho_r))):.2e})'
    )


# ---------------------------------------------------------------------------
# Test: batch over H (varying drive strength)
# ---------------------------------------------------------------------------


class TestBatchOverH:
    """Batch multiple Hamiltonians with shared jump operators."""

    def test_batch_H_matches_sequential(self):
        """Batched solve over H gives same results as sequential solves."""
        epsilons = [0.1, 0.3, 0.5]
        kappa = 1.0
        a = dq.destroy(N)

        # Build batched H: stack multiple Hamiltonians into shape (batch, n, n)
        Hs = [eps * (a + a.dag()) for eps in epsilons]
        H_batched = dq.stack(Hs)
        Ls = [jnp.sqrt(kappa) * a]

        # Batched solve
        result = dq.steadystate(
            H_batched, Ls, tol=TOL, max_iteration=MAX_ITER, krylov_size=KRYLOV
        )

        # Check output shape
        assert result.rho.shape == (len(epsilons), N, N), (
            f'Expected shape {(len(epsilons), N, N)}, got {result.rho.shape}'
        )

        # Check each slice matches a sequential solve
        for i, eps in enumerate(epsilons):
            H_i, Ls_i = _build_driven_damped_oscillator(N, eps, kappa)
            ref = _solve_single(H_i, Ls_i)
            _assert_rhos_close(result.rho, ref.rho, i)


# ---------------------------------------------------------------------------
# Test: batch over Ls (varying decay rate)
# ---------------------------------------------------------------------------


class TestBatchOverLs:
    """Batch multiple jump operators with a shared Hamiltonian."""

    def test_batch_Ls_matches_sequential(self):
        """Batched solve over Ls gives same results as sequential solves."""
        epsilon = 0.2
        kappas = [0.5, 1.0, 2.0]
        a = dq.destroy(N)

        H = epsilon * (a + a.dag())
        # Stack jump operators: shape (batch, n, n)
        Ls_batched = [dq.stack([jnp.sqrt(k) * a for k in kappas])]

        result = dq.steadystate(
            H, Ls_batched, tol=TOL, max_iteration=MAX_ITER, krylov_size=KRYLOV
        )

        assert result.rho.shape == (len(kappas), N, N), (
            f'Expected shape {(len(kappas), N, N)}, got {result.rho.shape}'
        )

        for i, k in enumerate(kappas):
            H_i, Ls_i = _build_driven_damped_oscillator(N, epsilon, k)
            ref = _solve_single(H_i, Ls_i)
            _assert_rhos_close(result.rho, ref.rho, i)


# ---------------------------------------------------------------------------
# Test: batch over both H and Ls (broadcast batching)
# ---------------------------------------------------------------------------


class TestBatchBroadcast:
    """Batch over H and Ls simultaneously with broadcasting."""

    def test_broadcast_H_and_Ls(self):
        """H of shape (3, n, n) and Ls of shape (3, n, n) broadcast element-wise."""
        from dynamiqs.options import Options

        epsilons = [0.1, 0.3, 0.5]
        kappas = [0.5, 1.0, 2.0]
        a = dq.destroy(N)

        Hs = [eps * (a + a.dag()) for eps in epsilons]
        H_batched = dq.stack(Hs)
        Ls_batched = [dq.stack([jnp.sqrt(k) * a for k in kappas])]

        # Explicit broadcast batching: pairs (H[i], Ls[i]) element-wise
        result = dq.steadystate(
            H_batched,
            Ls_batched,
            tol=TOL,
            max_iteration=MAX_ITER,
            krylov_size=KRYLOV,
            options=Options(cartesian_batching=False),
        )

        assert result.rho.shape == (len(epsilons), N, N), (
            f'Expected shape {(len(epsilons), N, N)}, got {result.rho.shape}'
        )

        # Each index i should match (epsilons[i], kappas[i])
        for i in range(len(epsilons)):
            H_i, Ls_i = _build_driven_damped_oscillator(N, epsilons[i], kappas[i])
            ref = _solve_single(H_i, Ls_i)
            _assert_rhos_close(result.rho, ref.rho, i)

    def test_cartesian_H_and_Ls(self):
        """H of shape (3, n, n) and Ls of shape (2, n, n) produce cartesian product."""
        epsilons = [0.1, 0.3, 0.5]
        kappas = [0.5, 2.0]
        a = dq.destroy(N)

        Hs = [eps * (a + a.dag()) for eps in epsilons]
        H_batched = dq.stack(Hs)
        Ls_batched = [dq.stack([jnp.sqrt(k) * a for k in kappas])]

        # Default cartesian batching: all (H[i], Ls[j]) combinations
        result = dq.steadystate(
            H_batched, Ls_batched, tol=TOL, max_iteration=MAX_ITER, krylov_size=KRYLOV
        )

        assert result.rho.shape == (len(epsilons), len(kappas), N, N), (
            f'Expected shape {(len(epsilons), len(kappas), N, N)}, '
            f'got {result.rho.shape}'
        )

        # Check each combination
        for i, eps in enumerate(epsilons):
            for j, k in enumerate(kappas):
                H_ij, Ls_ij = _build_driven_damped_oscillator(N, eps, k)
                ref = _solve_single(H_ij, Ls_ij)
                rho_slice = result.rho[i, j].to_jax()
                rho_ref = ref.rho.to_jax()
                assert jnp.allclose(rho_slice, rho_ref, atol=5e-3), (
                    f'Cartesian rho[{i},{j}] does not match sequential result '
                    f'(max diff = {float(jnp.max(jnp.abs(rho_slice - rho_ref))):.2e})'
                )

    def test_true_broadcast_stretching(self):
        """H shape (3, 1, n, n) and Ls shape (1, 2, n, n) broadcast to (3, 2, n, n).

        This tests real numpy-style broadcasting where dimensions of size 1 are
        stretched to match the other array:
            H batch shape:  (3, 1)  →  stretched to (3, 2)
            Ls batch shape: (1, 2)  →  stretched to (3, 2)
            result shape:   (3, 2, n, n)

        Entry [i, j] should correspond to (epsilons[i], kappas[j]).
        """
        from dynamiqs.options import Options

        epsilons = [0.1, 0.3, 0.5]
        kappas = [0.5, 2.0]
        a = dq.destroy(N)

        # H: shape (3, n, n) → reshape to (3, 1, n, n) via stack + reshape
        Hs = dq.stack([eps * (a + a.dag()) for eps in epsilons])
        H_batched = Hs.reshape(3, 1, N, N)  # (3, 1, n, n)

        # Ls: shape (2, n, n) → reshape to (1, 2, n, n)
        Ls_stacked = dq.stack([jnp.sqrt(k) * a for k in kappas])
        Ls_batched = [Ls_stacked.reshape(1, 2, N, N)]  # (1, 2, n, n)

        result = dq.steadystate(
            H_batched,
            Ls_batched,
            tol=TOL,
            max_iteration=MAX_ITER,
            krylov_size=KRYLOV,
            options=Options(cartesian_batching=False),
        )

        # broadcast_shapes((3, 1), (1, 2)) = (3, 2)
        assert result.rho.shape == (3, 2, N, N), (
            f'Expected shape (3, 2, {N}, {N}), got {result.rho.shape}'
        )

        # Entry [i, j] = solve(epsilons[i], kappas[j])
        for i, eps in enumerate(epsilons):
            for j, k in enumerate(kappas):
                H_ref, Ls_ref = _build_driven_damped_oscillator(N, eps, k)
                ref = _solve_single(H_ref, Ls_ref)
                rho_slice = result.rho[i, j].to_jax()
                rho_ref = ref.rho.to_jax()
                assert jnp.allclose(rho_slice, rho_ref, atol=5e-3), (
                    f'Broadcast rho[{i},{j}] does not match sequential result '
                    f'(max diff = {float(jnp.max(jnp.abs(rho_slice - rho_ref))):.2e})'
                )


# ---------------------------------------------------------------------------
# Test: batch over H with broadcasting (H batched, Ls scalar)
# ---------------------------------------------------------------------------


class TestBatchHScalarLs:
    """H is batched, Ls is not — Ls should be broadcast automatically."""

    def test_H_batched_Ls_scalar(self):
        """Non-batched Ls are broadcast to match batched H."""
        epsilons = [0.1, 0.2, 0.5]
        kappa = 1.0
        a = dq.destroy(N)

        Hs = [eps * (a + a.dag()) for eps in epsilons]
        H_batched = dq.stack(Hs)
        Ls = [jnp.sqrt(kappa) * a]  # not batched

        result = dq.steadystate(
            H_batched, Ls, tol=TOL, max_iteration=MAX_ITER, krylov_size=KRYLOV
        )

        assert result.rho.shape == (len(epsilons), N, N)
        assert jnp.all(result.infos.success)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
