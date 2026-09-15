import jax.numpy as jnp
import numpy as np
import pytest

import dynamiqs as dq

# with no jump operator the Lindblad equation is the Schrödinger equation, which the
# solvers must still integrate: summing an empty list of jump operators is the edge case


@pytest.fixture
def system():
    n = 6
    a = dq.destroy(n)  # `dia` layout, to also cover the sparse path
    H = 0.5 * (a + a.dag()) + a.dag() @ a
    return H, dq.fock(n, 0), jnp.linspace(0.0, 1.0, 5)


@pytest.mark.parametrize(
    ('method', 'kwargs'),
    [
        (dq.method.Tsit5(), {}),
        (dq.method.Tsit5(), {'assume_hermitian': False}),
        (dq.method.Tsit5(), {'vectorized': True}),
        (dq.method.Rouchon1(dt=1e-4), {}),
        (dq.method.Rouchon2(dt=1e-3), {}),
        (dq.method.Rouchon3(dt=1e-3), {}),
        (dq.method.Expm(), {}),
    ],
)
def test_mesolve_without_jump_ops_matches_sesolve(system, method, kwargs):
    H, psi0, tsave = system
    expected = dq.sesolve(H, psi0, tsave).states.todm().to_jax()

    states = dq.mesolve(H, [], psi0, tsave, method=method, **kwargs).states.to_jax()

    assert np.allclose(states, expected, atol=1e-4)


def test_mepropagator_without_jump_ops(system):
    H, _, tsave = system
    propagators = dq.mepropagator(H, [], tsave).propagators
    assert propagators.shape == (len(tsave), 36, 36)
