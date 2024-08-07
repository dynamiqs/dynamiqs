import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq


def rand_sesolve_args(n, nH, npsi0, nEs):
    kH, kpsi0, kEs = jax.random.split(jax.random.PRNGKey(42), 3)
    H = dq.random.herm(kH, (*nH, n, n))
    psi0 = dq.random.ket(kpsi0, (*npsi0, n, 1))
    Es = dq.random.complex(kEs, (nEs, n, n))
    return H, psi0, Es


@pytest.mark.parametrize('nH', [(), (3,), (3, 4)])
@pytest.mark.parametrize('npsi0', [(), (5,)])
def test_cartesian_batching(nH, npsi0):
    n = 2
    nEs = 7
    ntsave = 11

    # run sesolve
    H, psi0, Es = rand_sesolve_args(n, nH, npsi0, nEs)
    tsave = jnp.linspace(0, 0.01, ntsave)
    result = dq.sesolve(H, psi0, tsave, exp_ops=Es)

    # check result shape
    assert result.states.shape == (*nH, *npsi0, ntsave, n, 1)
    assert result.expects.shape == (*nH, *npsi0, nEs, ntsave)


# H has fixed shape (3, 4, n, n) for the next test case, we test a broad ensemble of
# compatible broadcastable shape
@pytest.mark.parametrize('npsi0', [(), (1,), (4,), (3, 1), (3, 4), (5, 1, 4)])
def test_flat_batching(npsi0):
    n = 2
    nH = (3, 4)
    nEs = 6
    ntsave = 11

    # run sesolve
    H, psi0, Es = rand_sesolve_args(n, nH, npsi0, nEs)
    tsave = jnp.linspace(0, 0.01, ntsave)
    options = dq.Options(cartesian_batching=False)
    result = dq.sesolve(H, psi0, tsave, exp_ops=Es, options=options)

    # check result shape
    broadcast_shape = jnp.broadcast_shapes(nH, npsi0)
    assert result.states.shape == (*broadcast_shape, ntsave, n, 1)
    assert result.expects.shape == (*broadcast_shape, nEs, ntsave)


def test_timearray_batching():
    # generic arrays
    a = dq.destroy(4)
    H0 = a + dq.dag(a)
    psi0 = dq.basis(4, 0)
    times = jnp.linspace(0.0, 1.0, 11)

    # == constant time array
    H_cte = jnp.stack([H0, 2 * H0])

    result = dq.sesolve(H_cte, psi0, times)
    assert result.states.shape == (2, 11, 4, 1)
    result = dq.sesolve(H0 + H_cte, psi0, times)
    assert result.states.shape == (2, 11, 4, 1)

    # == pwc time array
    values = jnp.arange(3 * 10).reshape(3, 10)
    H_pwc = dq.pwc(times, values, H0)

    result = dq.sesolve(H_pwc, psi0, times)
    assert result.states.shape == (3, 11, 4, 1)
    result = dq.sesolve(H0 + H_pwc, psi0, times)
    assert result.states.shape == (3, 11, 4, 1)

    # == modulated time array
    deltas = jnp.linspace(0.0, 1.0, 4)
    H_mod = dq.modulated(lambda t: jnp.cos(t * deltas), H0)

    result = dq.sesolve(H_mod, psi0, times)
    assert result.states.shape == (4, 11, 4, 1)
    result = dq.sesolve(H0 + H_mod, psi0, times)
    assert result.states.shape == (4, 11, 4, 1)

    # == callable time array
    omegas = jnp.linspace(0.0, 1.0, 5)
    H_cal = dq.timecallable(lambda t: jnp.cos(t * omegas[..., None, None]) * H0)

    result = dq.sesolve(H_cal, psi0, times)
    assert result.states.shape == (5, 11, 4, 1)
    result = dq.sesolve(H0 + H_cal, psi0, times)
    assert result.states.shape == (5, 11, 4, 1)


def test_sum_batching():
    a = dq.destroy(3)
    omegas = jnp.linspace(0, 2 * jnp.pi, 5)

    # some batched constant Hamiltonian of shape (5, 3, 3)
    H0 = omegas[..., None, None] * dq.dag(a) @ a

    # some batched modulated Hamiltonian of shape (5, 3, 3)
    H1 = dq.modulated(lambda t: jnp.cos(omegas * t), a + dq.dag(a))

    # sum of both Hamiltonians, also of shape (5, 3, 3)
    H = H0 + H1

    # run sesolve
    psi0 = dq.fock(3, 0)
    tsave = jnp.linspace(0, 2 * jnp.pi, 100)
    result = dq.sesolve(H, psi0, tsave)

    assert result.states.shape == (5, 100, 3, 1)
