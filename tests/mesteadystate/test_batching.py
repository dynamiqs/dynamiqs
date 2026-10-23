import jax
import pytest

import dynamiqs as dq

from ..order import TEST_LONG

# few steps, so that a stop that never fires fails fast
METHOD = dq.method.DiffraxSteadyState(dq.method.Kvaerno5(max_steps=1000))


def rand_mesteadystate_args(n, nH, nLs, nrho0, nEs):
    nkeys = len(nLs) + 3
    kH, *kLs, krho0, kEs = jax.random.split(jax.random.PRNGKey(42), nkeys)
    H = dq.random.operator(kH, n, batch=nH)
    Ls = [
        dq.random.operator(kL, n, hermitian=False, batch=nL)
        for kL, nL in zip(kLs, nLs, strict=True)
    ]
    rho0 = dq.random.dm(krho0, n, batch=nrho0)
    Es = dq.random.operator(kEs, n, batch=nEs)
    return H, Ls, rho0, Es


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('nrho0', [(), (4,)])
@pytest.mark.parametrize('nL1', [(), (5, 6)])
def test_cartesian_batching(nrho0, nL1):
    n = 2
    nH = (3,)
    nLs = [nL1, ()]
    nEs = 7

    H, Ls, rho0, Es = rand_mesteadystate_args(n, nH, nLs, nrho0, nEs)
    result = dq.mesteadystate(H, Ls, rho0=rho0, exp_ops=Es, method=METHOD)

    assert result.state.shape == (*nH, *nL1, *nrho0, n, n)
    assert result.expects.shape == (*nH, *nL1, *nrho0, nEs)
    assert result.residual.shape == (*nH, *nL1, *nrho0)
    assert result.infos.converged.shape == (*nH, *nL1, *nrho0)


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('nrho0', [(), (4,), (3, 1)])
def test_flat_batching(nrho0):
    n = 2
    nH = (3, 4)
    nEs = 7

    H, Ls, rho0, Es = rand_mesteadystate_args(n, nH, [()], nrho0, nEs)
    result = dq.mesteadystate(
        H, Ls, rho0=rho0, exp_ops=Es, method=METHOD, cartesian_batching=False
    )

    assert result.state.shape == (*nH, n, n)
    assert result.expects.shape == (*nH, nEs)
