import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.gradient import BackwardCheckpointed, Direct, Forward
from dynamiqs.integrators.core.low_rank_integrator import expval_from_m
from dynamiqs.method import LinearSolver, LowRank, Tsit5

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems import dense_ocavity, dia_ocavity, otdqubit

# we only test Tsit5 to keep the unit test suite fast


# use double precision for gradients
@pytest.fixture(scope='module', autouse=True)
def _double_precision():
    # keep precision changes local to this module to avoid cross-test leakage.
    prev_x64 = jax.config.read('jax_enable_x64')
    dq.set_precision('double')  # needed for time dependent test
    yield
    dq.set_precision('double' if prev_x64 else 'single')


def _lowrank_method(system, linear_solver=LinearSolver.QR):
    rank = 2 if system is otdqubit else system.n // 2
    return LowRank(
        rank=rank,
        ode_method=Tsit5(),
        linear_solver=linear_solver,
        key=jax.random.PRNGKey(0),
    )


@pytest.mark.run(order=TEST_LONG)
class TestMESolveAdaptiveLowRank(IntegratorTester):
    def test_key_is_required(self):
        with pytest.raises(TypeError):
            LowRank(rank=2, ode_method=Tsit5())

    @pytest.mark.parametrize('linear_solver', [LinearSolver.QR, LinearSolver.CHOLESKY])
    @pytest.mark.parametrize('system', [dense_ocavity, dia_ocavity, otdqubit])
    def test_correctness(self, system, linear_solver):
        self._test_correctness(system, _lowrank_method(system, linear_solver))

    @pytest.mark.parametrize('linear_solver', [LinearSolver.QR, LinearSolver.CHOLESKY])
    @pytest.mark.parametrize('system', [dense_ocavity, dia_ocavity, otdqubit])
    @pytest.mark.parametrize('gradient', [Direct(), BackwardCheckpointed(), Forward()])
    def test_gradient(self, system, gradient, linear_solver):
        self._test_gradient(system, _lowrank_method(system, linear_solver), gradient)

    @pytest.mark.parametrize('system', [dense_ocavity])
    def test_lowrank_states(self, system):
        result = system.run(_lowrank_method(system))
        assert isinstance(result, dq.MESolveLowRankResult)

        m = result.lowrank_states.to_jax()
        rho = m @ m.conj().swapaxes(-2, -1)
        assert jnp.allclose(result.states.to_jax(), rho)

    @pytest.mark.parametrize('system', [dense_ocavity])
    def test_save_extra_low_rank(self, system):
        rank = system.n // 2
        method = LowRank(
            rank=rank,
            ode_method=Tsit5(),
            key=jax.random.PRNGKey(0),
            is_save_extra_low_rank=True,
        )
        result = system.run(method, save_extra=lambda m: m)
        assert result.extra.shape[-2:] == (system.n, rank)

    @pytest.mark.parametrize('system', [dense_ocavity])
    def test_save_extra_full_rank_by_default(self, system):
        method = LowRank(
            rank=system.n // 2, ode_method=Tsit5(), key=jax.random.PRNGKey(0)
        )
        result = system.run(method, save_extra=lambda rho: rho)
        assert result.extra.shape[-2:] == (system.n, system.n)


def test_expval_from_m_accepts_qarray():
    # expval_from_m must accept a QArray operator directly (e.g. dia layout)
    # without densifying it first
    n, rank = 4, 2
    op = dq.number(n, layout=dq.dia)
    m = jax.random.normal(jax.random.PRNGKey(0), (n, rank), dtype=jnp.complex128)

    expect = jnp.sum(jnp.conj(m) * (op.to_jax() @ m))
    assert jnp.allclose(expval_from_m(m, op), expect)
