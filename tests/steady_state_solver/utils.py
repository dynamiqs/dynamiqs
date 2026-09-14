import jax.numpy as jnp

import dynamiqs as dq


def lindbladian_residual(H, Ls, rho):
    """Compute max-abs norm of L(rho), i.e. max|lindbladian(rho)|."""
    L_rho = dq.lindbladian(H, Ls, rho)
    return float(jnp.max(jnp.abs(L_rho.to_jax())))


def _dm_atol(precision):
    """Return the absolute tolerance for density matrix checks."""
    return 1e-8 if precision == 'double' else 1e-4


def assert_valid_dm(rho, precision):
    """Check that rho is a valid density matrix (hermitian, trace 1, PSD)."""
    atol = _dm_atol(precision)
    rho_jax = rho.to_jax()
    assert jnp.allclose(rho_jax, rho_jax.conj().T, atol=atol), 'rho is not Hermitian'
    assert jnp.isclose(jnp.trace(rho_jax), 1.0, atol=atol), (
        f'Tr(rho) = {jnp.trace(rho_jax)}, expected 1.0'
    )
    eigvals = jnp.linalg.eigvalsh(rho_jax)
    assert jnp.all(eigvals > -atol), (
        f'rho has negative eigenvalue: {float(jnp.min(eigvals))}'
    )


def simple_oscillator_analytical_steady_state(n, epsilon_a, kappa):
    """Analytical steady state for H=epsilon_a*(a+a^dagger), L=sqrt(kappa)*a."""
    alpha_ss = -2j * epsilon_a / kappa
    return dq.coherent_dm(n, alpha_ss)
