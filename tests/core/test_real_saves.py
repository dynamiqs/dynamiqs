import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.integrators.core import diffrax_integrator

from ..order import TEST_SHORT


def _solve():
    # a batch of driven, damped cavities, with expectation values and states saved
    a = dq.destroy(6)

    def solve(amplitude):
        H = a.dag() @ a + dq.modulated(lambda t: amplitude * jnp.cos(t), a + a.dag())
        result = dq.mesolve(
            H,
            [0.5 * a],
            dq.coherent(6, 0.5),
            jnp.linspace(0.0, 1.0, 4),
            exp_ops=[a.dag() @ a, a],
            progress_meter=False,
        )
        return result.expects, result.states.to_jax(), result.final_state.to_jax()

    return jax.vmap(solve)(jnp.array([0.5, 1.0, 1.5]))


@pytest.mark.run(order=TEST_SHORT)
def test_real_saves_are_exact(monkeypatch):
    # the saves are split into real arrays on GPU only, so CI forces it here
    reference = _solve()
    monkeypatch.setattr(diffrax_integrator, '_split_complex_saves', lambda: True)
    jax.clear_caches()
    split = _solve()
    for x, y in zip(split, reference, strict=True):
        assert x.dtype == y.dtype
        assert jnp.array_equal(x, y)
