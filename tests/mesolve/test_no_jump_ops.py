import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.method import Kvaerno5, Tsit5


@pytest.mark.parametrize('layout', [dq.dense, dq.dia])
@pytest.mark.parametrize('assume_hermitian', [True, False])
@pytest.mark.parametrize('method', [Tsit5(), Kvaerno5()])
@pytest.mark.parametrize('gpu_paths', [False, True])
def test_no_jump_ops(layout, assume_hermitian, method, gpu_paths, monkeypatch):
    # with no jump operators, mesolve evolves |psi><psi| as sesolve evolves |psi>
    if gpu_paths:
        # the GPU paths (fused DIA Lindbladian, unrolled stages): CI runs on CPU
        monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    jax.clear_caches()
    a = dq.destroy(8, layout=layout)
    H = a.dag() @ a + 0.3 * (a + a.dag())
    psi0 = dq.fock(8, 1)
    tsave = jnp.linspace(0.0, 1.0, 5)

    me = dq.mesolve(
        H,
        [],
        psi0.todm(),
        tsave,
        method=method,
        progress_meter=False,
        assume_hermitian=assume_hermitian,
    )
    se = dq.sesolve(H, psi0, tsave, method=method, progress_meter=False)

    assert jnp.allclose(me.states.to_jax(), se.states.todm().to_jax(), atol=1e-4)
    jax.clear_caches()
