import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.gradient import BackwardCheckpointed, Direct, Forward
from dynamiqs.method import Kvaerno5, Tsit5

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems import (
    dense_nojump_cavity,
    dense_ocavity,
    dia_nojump_cavity,
    dia_ocavity,
    otdqubit,
)

# we only test Tsit5 to keep the unit test suite fast, and Kvaerno5 for the vector
# field used by implicit methods


@pytest.mark.run(order=TEST_LONG)
class TestMESolveAdaptive(IntegratorTester):
    @pytest.mark.parametrize('system', [dense_ocavity, dia_ocavity, otdqubit])
    @pytest.mark.parametrize(
        ('vectorized', 'assume_hermitian'),
        [(True, True), (False, True), (False, False)],
    )
    def test_correctness(self, system, vectorized, assume_hermitian):
        self._test_correctness(
            system, Tsit5(), vectorized=vectorized, assume_hermitian=assume_hermitian
        )

    @pytest.mark.parametrize('system', [dense_ocavity, dia_ocavity, otdqubit])
    @pytest.mark.parametrize('gradient', [Direct(), BackwardCheckpointed(), Forward()])
    def test_gradient(self, system, gradient):
        self._test_gradient(system, Tsit5(), gradient)

    # otdqubit: the time-dependent drive makes Kvaerno5's root find fail on some steps
    @pytest.mark.parametrize('system', [dense_ocavity, otdqubit])
    def test_correctness_implicit(self, system):
        self._test_correctness(system, Kvaerno5())

    def test_correctness_no_jump_ops(self):
        self._test_correctness(dense_nojump_cavity, Tsit5())


@pytest.mark.run(order=TEST_LONG)
class TestMESolveAdaptiveGPUPaths(IntegratorTester):
    # on GPU, the Runge-Kutta stages are unrolled and the DIA Lindbladian is fused:
    # CI runs on CPU, so these tests force both
    @pytest.fixture(autouse=True)
    def gpu_paths(self, monkeypatch):
        monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
        jax.clear_caches()
        yield
        jax.clear_caches()

    @pytest.mark.parametrize('system', [dense_ocavity, dia_ocavity])
    def test_correctness(self, system):
        self._test_correctness(system, Tsit5())

    def test_gradient(self):
        # forward mode is covered by tests/core/test_unrolled_solvers.py
        self._test_gradient(dia_ocavity, Tsit5(), BackwardCheckpointed())

    def test_correctness_implicit(self):
        self._test_correctness(dia_ocavity, Kvaerno5())

    def test_correctness_no_jump_ops(self):
        self._test_correctness(dia_nojump_cavity, Tsit5())

    @pytest.mark.parametrize('layout', [dq.dense, dq.dia])
    def test_correctness_time_dependent_jump_ops(self, layout):
        # jump operators made of constant, modulated and pwc terms, clipped or summed,
        # give the same states as the same operators given as callables
        n = 8
        a, eye = dq.destroy(n, layout=layout), dq.eye(n, layout=layout)
        Ls = [
            0.7 * (a @ a - dq.modulated(lambda t: 0.5 * jnp.exp(1j * t), eye)),
            dq.pwc([0.0, 0.5, 1.2], [0.3, 1.0j], a),
            dq.constant(0.2 * a.dag() @ a).clip(0.3, 1.5),
            (0.5 * a.dag() - dq.modulated(jnp.sin, a)).clip(0.4, 1.7),
        ]
        Ls_callable = [
            dq.timecallable(L.__call__, discontinuity_ts=L.discontinuity_ts) for L in Ls
        ]
        H, rho0, tsave = a.dag() @ a, dq.coherent_dm(n, 1.0), jnp.linspace(0, 2, 3)
        states = dq.mesolve(H, Ls, rho0, tsave).states.to_jax()
        expected = dq.mesolve(H, Ls_callable, rho0, tsave).states.to_jax()
        assert jnp.allclose(states, expected, atol=1e-6)
