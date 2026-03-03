import jax
import jax.numpy as jnp

import dynamiqs as dq


def build_random_single_mode(n: int, seed: int = 0, gamma: float = 0.1):
    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)

    H_real = jax.random.normal(key1, (n, n))
    H_imag = jax.random.normal(key2, (n, n))
    H_complex = H_real + 1j * H_imag
    H_herm = (H_complex + H_complex.conj().T) / 2
    H = dq.asqarray(H_herm)

    L_real = jax.random.normal(key1, (n, n))
    L_imag = jax.random.normal(key2, (n, n))
    L_complex = (L_real + 1j * L_imag) * gamma
    L = dq.asqarray(L_complex)

    Ls = [L]
    return H, Ls


to_rad_MHz = 2 * jnp.pi * 1e-3  # Converts units in rad.MHz


def eps_d_from_na(na: int, g2: float) -> float:
    r"""Choose a suitable drive parameter eps_d to keep the truncature large enough.

    The optimal truncature is chosen under the condition:
    $$ n_a > |\alpha|^2 + 15|\alpha| $$
    with
    $$ \alpha = \sqrt{\varepsilon_d / g_2} $$

    Args:
        na (int): Truncature of the memory mode.
        g2 (float): Coupling of memory and buffer mode.

    Returns:
        float: Drive on the buffer mode.
    """
    eps_map = {12: 1.0, 24: 2.0, 32: 7.0, 46: 12.0}

    if na in eps_map:
        return eps_map[na]

    n_target = na - 1
    alpha = (-15.0 + jnp.sqrt(225.0 + 4.0 * n_target)) / 2.0
    return float(g2 * alpha**2)


def build_two_modes(
    n_a: int,
    n_b: int,
    g2: float = 2,
    eps_d: float | None = None,
    kappa_b: float = 8,
    kappa_a: float = 1,
):
    """Build the generators for the Hamiltonian of a memory-buffer cat qubit.

    Args:
        n_a: Truncature of the memory mode.
        n_b: Truncature of the buffer mode.
        g2: Coupling strength between the two modes.
        eps_d: Drive on the buffer mode.
        kappa_a: Single photon loss on the memory mode.
        kappa_b: Single photon loss on the buffer mode.

    Returns:
        tuple: Hamiltonian and list of jump operators.
    """
    if eps_d is None:
        eps_d = eps_d_from_na(n_a, g2)

    g2 = g2 * to_rad_MHz
    kappa_b = kappa_b * to_rad_MHz
    kappa_a = kappa_a * to_rad_MHz
    eps_d = eps_d * to_rad_MHz

    a, b = dq.destroy(n_a, n_b)
    H0 = g2 * (a @ a @ b.dag() + a.dag() @ a.dag() @ b) + eps_d * (b + b.dag())

    Ls = [dq.asqarray(jnp.sqrt(kappa_b) * b), dq.asqarray(jnp.sqrt(kappa_a) * a)]

    return H0, Ls
