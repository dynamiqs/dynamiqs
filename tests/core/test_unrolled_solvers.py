import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.integrators.core import unrolled_solvers

from ..order import TEST_SHORT


def _solve(gradient=None):
    # a driven, damped cavity with a pulse edge (a jump time), and its derivative
    # with respect to the drive amplitude
    a = dq.destroy(8)
    tsave = jnp.linspace(0.0, 2.0, 11)

    def final_population(amplitude):
        H = a.dag() @ a + dq.modulated(
            lambda t: amplitude * jnp.where(t < 0.7, 1.0, 0.3),
            a + a.dag(),
            discontinuity_ts=[0.7],
        )
        result = dq.mesolve(
            H,
            [0.5 * a],
            dq.fock(8, 0),
            tsave,
            exp_ops=[a.dag() @ a],
            gradient=gradient,
            progress_meter=False,
        )
        return result.expects[0, -1].real, result.infos.nsteps

    return final_population


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('unroll', [False, True])
def test_unrolled_matches_diffrax(monkeypatch, unroll):
    # the stages are unrolled on GPU only, so CI forces them here
    reference, reference_steps = _solve()(1.3)
    monkeypatch.setattr(unrolled_solvers, '_unroll_stages', lambda: unroll)
    jax.clear_caches()
    value, steps = _solve()(1.3)
    assert steps == reference_steps
    assert jnp.allclose(value, reference, rtol=1e-5)
    grad = jax.jacfwd(lambda x: _solve(dq.gradient.Forward())(x)[0])(1.3)
    jax.clear_caches()
    monkeypatch.setattr(unrolled_solvers, '_unroll_stages', lambda: False)
    grad_ref = jax.jacfwd(lambda x: _solve(dq.gradient.Forward())(x)[0])(1.3)
    assert jnp.allclose(grad, grad_ref, rtol=1e-4)
