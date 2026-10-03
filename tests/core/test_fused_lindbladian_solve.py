import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.integrators.core import diffrax_integrator

from ..order import TEST_SHORT


def _final_state():
    # a driven, damped cavity, all its operators in DIA format
    a = dq.destroy(12)
    H = a.dag() @ a + dq.modulated(lambda t: 0.8 * jnp.cos(3.0 * t), a + a.dag())
    result = dq.mesolve(
        H,
        [0.5 * a, 0.2 * a.dag() @ a],
        dq.fock(12, 0),
        jnp.linspace(0.0, 2.0, 5),
        progress_meter=False,
    )
    return result.final_state.to_jax(), result.infos.nsteps


@pytest.mark.run(order=TEST_SHORT)
def test_fused_lindbladian_in_mesolve(monkeypatch):
    # mesolve fuses the DIA Lindbladian on GPU only, so CI forces it here
    reference, reference_steps = _final_state()
    calls = []
    fused_terms = diffrax_integrator.lindbladian_sparsedia_terms

    def spy(*args):
        calls.append(None)
        return fused_terms(*args)

    monkeypatch.setattr(diffrax_integrator, 'lindbladian_sparsedia_terms', spy)
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    jax.clear_caches()
    fused, steps = _final_state()
    assert calls  # the solve took the fused path
    assert steps == reference_steps
    assert jnp.allclose(fused, reference, atol=1e-5)  # single precision
    assert jnp.array_equal(fused, fused.mT.conj())  # exactly Hermitian


@pytest.mark.run(order=TEST_SHORT)
def test_wide_band_is_not_fused(monkeypatch):
    # rows of rho 2 * 1900 apart (n = 2048) do not fit in an A100's L2: mesolve uses
    # the products instead of the fused Lindbladian (traced only, not solved)
    calls = []

    def spy(*args):
        calls.append(None)

    monkeypatch.setattr(diffrax_integrator, 'lindbladian_sparsedia_terms', spy)
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    n = 2048
    H = dq.sparsedia_from_dict({-1900: jnp.ones(n - 1900), 1900: jnp.ones(n - 1900)})

    def final_state():
        L, psi0, tsave = dq.destroy(n), dq.fock(n, 0), jnp.linspace(0.0, 1.0, 2)
        result = dq.mesolve(H, [L], psi0, tsave, progress_meter=False)
        return result.final_state.to_jax()

    jax.eval_shape(final_state)
    assert not calls
