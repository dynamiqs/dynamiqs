import jax.numpy as jnp
import jax.random

import dynamiqs as dq
from dynamiqs import (
    Options,
    mcsolve,
    mesolve,
)


def test_against_mesolve(ysave_atol=1e-1):
    num_traj = 161
    options = Options(progress_meter=None)
    dim = 4
    a = dq.destroy(dim)
    y0 = dq.basis(dim, 0)
    jump_ops = [0.1 * a]
    exp_ops = [dq.dag(a) @ a]
    H0 = 0.1 * dq.dag(a) @ a + 0.05 * (a + dq.dag(a))
    tsave = jnp.linspace(0.0, 100.0, 41)

    mcresult = mcsolve(
        H0,
        jump_ops,
        y0,
        tsave,
        keys=jax.random.split(jax.random.key(31), num=num_traj),
        exp_ops=exp_ops,
        options=options,
        root_finder=None,
    )
    meresult = mesolve(
        H0,
        jump_ops,
        y0,
        tsave,
        exp_ops=exp_ops,
        options=options,
    )
    errs = jnp.linalg.norm(meresult.expects - mcresult.expects, axis=(-2, -1))
    assert jnp.all(errs <= ysave_atol)
