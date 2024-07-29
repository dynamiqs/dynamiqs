import jax.numpy as jnp
import numpy as np

import dynamiqs as dq


# test for the method coherent
def test_coherent():
    alpha1, alpha2 = 1.0, 1.0j
    alphas1, alphas2 = np.linspace(0, 1, 5), np.linspace(0, 1, 7) * 1j
    n1, n2 = 8, 8

    # Short tensor product
    state1 = dq.tensor(dq.coherent(n1, alpha1), dq.coherent(n2, alpha2))
    state2 = dq.coherent((n1, n2), (alpha1, alpha2))
    assert np.allclose(state1, state2)

    # Short batching
    state1 = dq.coherent(n1, alphas1)
    state2 = jnp.stack([dq.coherent(n1, alpha) for alpha in alphas1])
    assert np.allclose(state1, state2)

    # Short batching + tensor product
    state1 = dq.tensor(dq.coherent(n1, alphas1), dq.coherent(n2, alpha2)[None, ...])
    state2 = dq.coherent((n1, n2), (alphas1, alpha2))
    assert np.allclose(state1, state2)

    # Double short batching + tensor product
    state1 = dq.tensor(
        dq.coherent(n1, alphas1)[None, ...], dq.coherent(n2, alphas2)[:, None, ...]
    )
    state2 = dq.coherent((n1, n2), (alphas1, alphas2[:, None]))
    assert np.allclose(state1, state2)

    print()

    # Double short batching + tensor product with single full array
    state1 = dq.tensor(
        dq.coherent(n1, alphas1)[None, ...], dq.coherent(n2, alphas2)[:, None, ...]
    )
    state2 = dq.coherent((n1, n2), alphas1 * alphas2[:, None])
    assert np.allclose(state1, state2)
