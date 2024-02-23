import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq


@pytest.mark.parametrize('cartesian_batching', [True, False])
def test_batching(cartesian_batching):  # noqa: PLR0915
    n = 8
    nt = 11
    nH = 3
    nL1 = 4 if cartesian_batching else nH
    nL2 = 5 if cartesian_batching else nH
    npsi0 = 6 if cartesian_batching else nH
    nEs = 7

    options = dq.options.Options(cartesian_batching=cartesian_batching)

    # create random objects
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(42), 5)
    H = dq.rand_herm(k1, (nH, n, n))
    Ls = [dq.rand_herm(k2, (nL1, n, n)), dq.rand_herm(k3, (nL2, n, n))]
    exp_ops = dq.rand_complex(k4, (nEs, n, n))
    psi0 = dq.rand_ket(k5, (npsi0, n, 1))
    tsave = jnp.linspace(0, 0.01, nt)

    # no batching
    Ls0 = [L[0] for L in Ls]
    result = dq.mesolve(H[0], Ls0, psi0[0], tsave, exp_ops=exp_ops, options=options)
    assert result.states.shape == (nt, n, n)
    assert result.expects.shape == (nEs, nt)

    # H batched
    result = dq.mesolve(H, Ls0, psi0[0], tsave, exp_ops=exp_ops, options=options)
    assert result.states.shape == (nH, nt, n, n)
    assert result.expects.shape == (nH, nEs, nt)

    # Ls batched
    result = dq.mesolve(H[0], Ls, psi0[0], tsave, exp_ops=exp_ops, options=options)
    if cartesian_batching:
        assert result.states.shape == (nL1, nL2, nt, n, n)
        assert result.expects.shape == (nL1, nL2, nEs, nt)
    else:
        assert result.states.shape == (nH, nt, n, n)
        assert result.expects.shape == (nH, nEs, nt)

    # psi0 batched
    result = dq.mesolve(H[0], Ls0, psi0, tsave, exp_ops=exp_ops, options=options)
    assert result.states.shape == (npsi0, nt, n, n)
    assert result.expects.shape == (npsi0, nEs, nt)

    # H and Ls batched
    result = dq.mesolve(H, Ls, psi0[0], tsave, exp_ops=exp_ops, options=options)
    if cartesian_batching:
        assert result.states.shape == (nH, nL1, nL2, nt, n, n)
        assert result.expects.shape == (nH, nL1, nL2, nEs, nt)
    else:
        assert result.states.shape == (nH, nt, n, n)
        assert result.expects.shape == (nH, nEs, nt)

    # H and psi0 batched
    result = dq.mesolve(H, Ls0, psi0, tsave, exp_ops=exp_ops, options=options)
    if cartesian_batching:
        assert result.states.shape == (nH, npsi0, nt, n, n)
        assert result.expects.shape == (nH, npsi0, nEs, nt)
    else:
        assert result.states.shape == (nH, nt, n, n)
        assert result.expects.shape == (nH, nEs, nt)

    # H, Ls and psi0 batched
    result = dq.mesolve(H, Ls, psi0, tsave, exp_ops=exp_ops, options=options)
    if cartesian_batching:
        assert result.states.shape == (nH, nL1, nL2, npsi0, nt, n, n)
        assert result.expects.shape == (nH, nL1, nL2, npsi0, nEs, nt)
    else:
        assert result.states.shape == (nH, nt, n, n)
        assert result.expects.shape == (nH, nEs, nt)
