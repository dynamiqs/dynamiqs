import jax
import jax.numpy as jnp
import dynamiqs as dq
from math import pi

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


def build_single_mode(n_b: int):
    # Conversion: Value (MHz) * 1e-3 (to GHz) * 2pi (to angular rad/ns)
    to_rad_ns = 1e-3 * 2 * pi

    detuning_rad_ns = 0.1 * to_rad_ns
    drive_rad_ns = 0.2 * to_rad_ns
    kerr_4_rad_ns = -1.5 * to_rad_ns
    kerr_6_rad_ns = -0.3 * to_rad_ns
    kappa_rad_ns = 0.5 * to_rad_ns

    d = int(n_b)
    b = dq.destroy(d)  # type: ignore[assignment]

    H = (
        detuning_rad_ns * (b.dag() @ b)  # type: ignore[attr-defined]
        + drive_rad_ns * (b + b.dag())  # type: ignore[attr-defined]
        + kerr_4_rad_ns / 2 * (dq.powm(b.dag(), 2) @ dq.powm(b, 2))  # type: ignore[attr-defined]
        + kerr_6_rad_ns / 6 * (dq.powm(b.dag(), 3) @ dq.powm(b, 3))  # type: ignore[attr-defined]
    )

    # Dissipation: L = sqrt(kappa) * b
    Ls = [jnp.sqrt(kappa_rad_ns) * b]  # type: ignore[operator]

    return H, Ls


kHz, MHz = 2 * jnp.pi * 1e-3, 2 * jnp.pi
ns, us = 1e-3, 1e0


def eps_d_from_na(na: int, g2: float) -> float:
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
    if eps_d is None:
        eps_d = eps_d_from_na(n_a, g2)
    g2 = g2 * MHz
    kappa_b = kappa_b * MHz
    # kappa_a = 0.05 * MHz
    kappa_a = kappa_a * MHz
    eps_d = eps_d * MHz
    a, b = dq.destroy(n_a, n_b)
    adag = a.dag()
    bdag = b.dag()
    H0 = g2 * (a @ a @ bdag + adag @ adag @ b) + eps_d * (b + bdag)
    Ls = [dq.asqarray(jnp.sqrt(kappa_b) * b), dq.asqarray(jnp.sqrt(kappa_a) * a)]

    return H0, Ls
