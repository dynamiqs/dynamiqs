import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_INSTANT


@pytest.mark.run(order=TEST_INSTANT)
def test_tracing():
    # prepare inputs
    phase = jnp.array([0.5, 1.0, 1.5])
    phases = jnp.stack([phase, phase + 1.0])
    dim = 4
    alpha = 0.5
    alphas = jnp.array([0.5, 1.0, 1.5])

    # check that no error is raised while tracing the functions
    jax.jit(dq.snap_gate)(phase)
    jax.jit(dq.snap_gate)(phases)
    jax.jit(dq.cd_gate, static_argnums=(0,))(dim, alpha)
    jax.jit(dq.cd_gate, static_argnums=(0,))(dim, alphas)
