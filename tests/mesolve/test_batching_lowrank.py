import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_LONG


@pytest.fixture(scope='module', autouse=True)
def _double_precision():
    # Keep precision changes local to this module to avoid cross-test leakage.
    prev_x64 = jax.config.read('jax_enable_x64')
    dq.set_precision('double')  # needed for time dependent test
    yield
    dq.set_precision('double' if prev_x64 else 'single')


def rand_mesolve_args(n, nH, nLs, npsi0, nEs):
    nkeys = len(nLs) + 3
    kH, *kLs, kpsi0, kEs = jax.random.split(jax.random.PRNGKey(42), nkeys)
    H = dq.random.operator(kH, n, batch=nH)
    Ls = [dq.random.operator(kL, n, batch=nL) for kL, nL in zip(kLs, nLs, strict=True)]
    psi0 = dq.random.ket(kpsi0, n, batch=npsi0)
    Es = dq.random.operator(kEs, n, hermitian=False, batch=nEs)
    return H, Ls, psi0, Es


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('nH', [(), (3,), (3, 4)])
@pytest.mark.parametrize('npsi0', [(), (5,)])
@pytest.mark.parametrize('nL1', [(), (7, 8)])
@pytest.mark.parametrize('nL2', [(), (9,)])
def test_cartesian_batching_lowrank(nH, npsi0, nL1, nL2):
    n = 2
    nLs = [nL1, nL2]
    nEs = 10
    ntsave = 11

    # run mesolve
    H, Ls, psi0, Es = rand_mesolve_args(n, nH, nLs, npsi0, nEs)
    tsave = jnp.linspace(0, 0.01, ntsave)
    result = dq.mesolve(
        H, Ls, psi0, tsave, exp_ops=Es, method=dq.method.LowRank(rank=n)
    )

    # check result shape
    assert result.states.shape == (*nH, *nL1, *nL2, *npsi0, ntsave, n, n)
    assert result.lowrank_states.shape == (*nH, *nL1, *nL2, *npsi0, ntsave, n, n)
    assert result.expects.shape == (*nH, *nL1, *nL2, *npsi0, nEs, ntsave)


# H has fixed shape (3, 4, n, n) for the next test case, we test a broad ensemble of
# compatible broadcastable shape
@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('nL1', [(), (5, 1, 4)])
@pytest.mark.parametrize('npsi0', [(), (1,), (4,), (3, 1), (3, 4), (5, 1, 4)])
def test_flat_batching_lowrank(nL1, npsi0):
    n = 2
    nH = (3, 4)
    nLs = [nL1, ()]
    nEs = 6
    ntsave = 11

    # run mesolve
    H, Ls, psi0, Es = rand_mesolve_args(n, nH, nLs, npsi0, nEs)
    tsave = jnp.linspace(0, 0.01, ntsave)
    options = dq.Options(cartesian_batching=False)
    result = dq.mesolve(
        H,
        Ls,
        psi0,
        tsave,
        exp_ops=Es,
        method=dq.method.LowRank(rank=n),
        options=options,
    )

    # check result shape
    broadcast_shape = jnp.broadcast_shapes(nH, nL1, npsi0)
    assert result.states.shape == (*broadcast_shape, ntsave, n, n)
    assert result.lowrank_states.shape == (*broadcast_shape, ntsave, n, n)
    assert result.expects.shape == (*broadcast_shape, nEs, ntsave)


@pytest.mark.run(order=TEST_LONG)
def test_batching_boris_lowrank():
    n = 9
    a = dq.destroy(n)

    # first modulated operator (1, 5, 9, 9)
    omega1 = jnp.linspace(0, 1, 5)[None, :]
    f = lambda t: jnp.cos(omega1 * t)
    H1 = dq.modulated(f, a + a.dag())

    # second modulated operator 2 (7, 1, 9, 9)
    omega2 = jnp.linspace(0, 1, 7)[:, None]
    f = lambda t: jnp.cos(omega2 * t)
    H2 = dq.modulated(f, 1j * a - 1j * a.dag())

    # Hamiltonian
    H = H1 + H2

    rho0 = dq.fock_dm(n, range(3))  # (3, 9, 9)
    jump_ops = [a]
    result = dq.mesolve(
        H, jump_ops, rho0, [0, 1], method=dq.method.LowRank(rank=n // 2)
    )
    assert result.states.shape == (7, 5, 3, 2, 9, 9)
    assert result.lowrank_states.shape == (7, 5, 3, 2, 9, n // 2)
    assert result.tsave is not None
