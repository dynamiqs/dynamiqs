import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dynamiqs as dq
from dynamiqs.gradient import Direct
from dynamiqs.method import Expm

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems import dense_cavity


@pytest.mark.run(order=TEST_LONG)
class TestSESolveExpm(IntegratorTester):
    def test_correctness(self):
        self._test_correctness(dense_cavity, Expm())

    def test_gradient(self):
        self._test_gradient(dense_cavity, Expm(), Direct())

    @pytest.mark.parametrize('backend', ['cpu', 'gpu'])
    def test_reused_propagators(self, monkeypatch, backend):
        # save intervals of equal length reuse a propagator, except across a change of
        # H(t), which happens on save times (0.5, 2.0) and inside a save interval
        # (1.05); CI runs on CPU, so it also forces the GPU's batched path here
        monkeypatch.setattr(jax, 'default_backend', lambda: backend)
        jax.clear_caches()
        H0 = dq.random.herm(jax.random.PRNGKey(0), (4, 4))
        times, values = [0.0, 0.5, 1.05, 2.0, 3.0], [1.0, -0.5, 0.7, 2.0]
        H = dq.pwc(times, values, H0)
        psi0 = dq.random.ket(jax.random.PRNGKey(1), 4)
        tsave = jnp.linspace(0.1, 3.0, 30)
        result = dq.sesolve(H, psi0, tsave, method=Expm(), t0=0.0)

        # H(t) = c(t) H0 commutes with itself: psi(t) = exp(-i Phi(t) H0) psi0, with
        # Phi(t) the integral of c from 0 to t
        phis = np.interp(tsave, times, np.cumsum([0.0, *np.diff(times) * values]))
        H0 = H0.to_jax()
        expected = [
            jax.scipy.linalg.expm(-1j * phi * H0) @ psi0.to_jax() for phi in phis
        ]
        assert jnp.allclose(result.states.to_jax(), jnp.stack(expected), atol=1e-12)

    def test_composite_hamiltonian(self):
        # two cavities of detunings delta and 2 delta: psi(t) is the product of the
        # coherent states of amplitudes alpha0 e^(-i delta t) and alpha0 e^(-2i delta t)
        n, delta, alpha0, t = 8, 1.0, 0.5, 0.3
        N, I = dq.number(n), dq.eye(n)
        H_a = dq.tensor(N, I, composite=True)
        H_b = dq.tensor(I, 2 * N, composite=True)
        H = delta * (H_a + H_b)
        psi0 = dq.tensor(dq.coherent(n, alpha0), dq.coherent(n, alpha0))
        result = dq.sesolve(H, psi0, [0.0, t], method=Expm())

        alpha_a = alpha0 * jnp.exp(-1j * delta * t)
        alpha_b = alpha0 * jnp.exp(-2j * delta * t)
        expected = dq.tensor(dq.coherent(n, alpha_a), dq.coherent(n, alpha_b))
        assert jnp.allclose(result.states[-1].to_jax(), expected.to_jax(), atol=1e-6)
