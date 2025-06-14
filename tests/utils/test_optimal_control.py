import jax
import pytest

import dynamiqs as dq

from ..order import TEST_INSTANT


@pytest.mark.run(order=TEST_INSTANT)
def test_snap_gate():
    # prepare inputs
    phase = [0.5, 1.0, 1.5]
    phases = [phase, phase, phase]

    # check that no error is raised while tracing the function
    jax.jit(dq.snap_gate)(phase)
    jax.jit(dq.snap_gate)(phases)


@pytest.mark.run(order=TEST_INSTANT)
def test_cd_gate():
    # prepare inputs
    dim = 4
    alpha = 0.5
    alphas = [0.5, 1.0, 1.5]

    # check that no error is raised while tracing the function
    jax.jit(dq.cd_gate, static_argnums=(0,))(dim, alpha)
    jax.jit(dq.cd_gate, static_argnums=(0,))(dim, alphas)
