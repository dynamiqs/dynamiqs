import equinox as eqx
import jax.numpy as jnp

from ...options import Options
from ...gradient import Gradient
from ...result import FloquetResult, SEPropagatorResult, Saved
from ...solver import Solver, Tsit5
from ...time_array import TimeArray
from ...integrators.apis.sepropagator import sepropagator

__all__ = ['floquet']


def floquet(
    H: TimeArray,
    T: float,
    *,
    t: float = 0.0,
    floquet_result_0: FloquetResult | None = None,
    solver: Solver = Tsit5(),
    gradient: Gradient | None = None,
    options=Options(),
) -> FloquetResult:
    r"""Compute the Floquet modes $\Phi_{m}(t)$ and quasi energies $\epsilon_m$ for a
    periodically driven system. The Floquet modes at t=0 are defined by the eigenvalue
    equation
    $$
        U(0, T)\Phi_{m}(0) = \exp(-i \epsilon_{m} T)\Phi_{m}(0),
    $$
    where U(0, T) is the propagator from time 0 to time T, and T is the period
    of the drive.

    Warning:
        No check is made that the Hamiltonian is actually time periodic with the
        supplied period.

    The Floquet modes at other times t are computed via the relationship
    $$
        \Phi_{m}(t) = \exp(i\epsilon_{m}t)U(0, t)\Phi_{m}(0).
    $$

    Args:
        H _(array-like or time-array of shape (...H, n, n))_: Hamiltonian.
        T _(float)_: Period of the drive
        t _(float)_: Time at which to compute the Floquet modes
        floquet_result_0 _(FloquetResult)_: [`dq.FloquetResult`][dynamiqs.FloquetResult]
            object, containing the Floquet modes at t=0 (with associated quasienergies)
        solver: Solver for the integration.
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].

    Returns:
        [`dq.FloquetResult`][dynamiqs.FloquetResult] object holding the result of the
            Floquet computation. Use the attribute `floquet_modes` to access the saved
            Floquet modes, and the attribute `quasi_energies` the associated quasi
            energies, more details in [`dq.FloquetResult`][dynamiqs.FloquetResult].
    """
    t_mod = jnp.mod(t, T)
    if t_mod == 0.0 and floquet_result_0 is not None:
        return floquet_result_0
    elif t_mod == 0.0:
        return _floquet_0(H, T, solver=solver, gradient=gradient, options=options)
    elif floquet_result_0 is None:
        floquet_result_0 = _floquet_0(H, T, solver=solver, gradient=gradient, options=options)
    # otherwise, we have asked for t != 0 and have provided the Floquet modes at t = 0
    U_result = _floquet_propagator(H, t, solver=solver, gradient=gradient, options=options)
    U = U_result.propagators
    floquet_modes_0 = floquet_result_0.floquet_modes
    quasi_energies = floquet_result_0.quasi_energies
    # note that ordering of indices important here: for now, floquet modes
    # stored as row vectors so k indexes which floquet mode, j indexes its entries
    floquet_modes_t = jnp.einsum(
        "...ij,...kj,...k->...ki",
        U,
        floquet_modes_0,
        jnp.exp(1j * quasi_energies * t)
    )
    saved = Saved(floquet_modes_t, quasi_energies, None)
    return FloquetResult(
        U_result.tsave, solver, gradient, options, saved, U_result.infos
    )


def _floquet_propagator(
    H: TimeArray,
    t: float,
    *,
    solver: Solver = Tsit5(),
    gradient: Gradient | None = None,
    options=Options(),
) -> SEPropagatorResult:
    options = eqx.tree_at(
        lambda x: x.save_states, options, False, is_leaf=lambda x: x is None
    )
    return sepropagator(
        H, jnp.asarray([0, t]), solver=solver, gradient=gradient, options=options
    )


def _floquet_0(
    H: TimeArray,
    T: float,
    *,
    solver: Solver = Tsit5(),
    gradient: Gradient | None = None,
    options=Options(),
) -> FloquetResult:
    U_result = _floquet_propagator(H, T, solver=solver, gradient=gradient, options=options)
    evals, evecs = jnp.linalg.eig(U_result.propagators)
    # quasi energies are only defined modulo 2pi / T. Usual convention is to normalize
    # quasi energies to the region -pi/T, pi/T
    omega_d = 2.0 * jnp.pi / T
    quasi_es = jnp.angle(evals) / T
    quasi_es = jnp.mod(quasi_es, omega_d)
    quasi_es = jnp.where(quasi_es > 0.5 * omega_d, quasi_es - omega_d, quasi_es)
    # quasi_es = jnp.mod(quasi_es, 2.0 * jnp.pi / T)
    # quasi_es = jnp.where(quasi_es > jnp.pi / T, quasi_es - 2.0 * jnp.pi / T, quasi_es)
    evecs = jnp.swapaxes(evecs, axis1=-1, axis2=-2)[..., None]
    saved = Saved(evecs, quasi_es, None)
    return FloquetResult(
        U_result.tsave, solver, gradient, options, saved, U_result.infos
    )
