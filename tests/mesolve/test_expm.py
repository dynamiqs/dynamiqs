import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dynamiqs as dq
from dynamiqs.gradient import Direct
from dynamiqs.method import Expm

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems import dense_ocavity


@pytest.mark.run(order=TEST_LONG)
class TestMESolveExpm(IntegratorTester):
    def test_correctness(self):
        self._test_correctness(dense_ocavity, Expm())

    def test_gradient(self):
        self._test_gradient(dense_ocavity, Expm(), Direct())

    @pytest.mark.parametrize('backend', ['cpu', 'gpu'])
    def test_reused_propagators(self, monkeypatch, backend):
        # as in sesolve, with a Liouvillian: H(t) = c(t) H0 changes on save times (0.5,
        # 2.0) and inside a save interval (1.05); CI runs on CPU, so it also forces the
        # GPU's batched path here
        monkeypatch.setattr(jax, 'default_backend', lambda: backend)
        jax.clear_caches()
        H0 = dq.random.herm(jax.random.PRNGKey(0), (3, 3))
        times, values = [0.0, 0.5, 1.05, 2.0, 3.0], [1.0, -0.5, 0.7, 2.0]
        gamma = 0.3
        rho0 = dq.random.dm(jax.random.PRNGKey(1), 3)
        tsave = jnp.linspace(0.1, 3.0, 30)
        H = dq.pwc(times, values, H0)
        result = dq.mesolve(
            H, [jnp.sqrt(gamma) * H0], rho0, tsave, method=Expm(), t0=0.0
        )

        # with L = sqrt(gamma) H0, the eigenbasis of H0 diagonalizes the dynamics:
        # rho_jk(t) = rho_jk(0) exp(-i Phi(t) (E_j - E_k) - gamma t (E_j - E_k)^2 / 2),
        # with Phi(t) the integral of c from 0 to t
        energies, vectors = jnp.linalg.eigh(H0.to_jax())
        gaps = energies[:, None] - energies[None, :]
        rho0_eigen = vectors.conj().T @ rho0.to_jax() @ vectors
        phis = np.interp(tsave, times, np.cumsum([0.0, *np.diff(times) * values]))
        expected = [
            vectors
            @ (rho0_eigen * jnp.exp(-1j * phi * gaps - 0.5 * gamma * t * gaps**2))
            @ vectors.conj().T
            for phi, t in zip(phis, tsave, strict=True)
        ]
        assert jnp.allclose(result.states.to_jax(), jnp.stack(expected), atol=1e-12)
