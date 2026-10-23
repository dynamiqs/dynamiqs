import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.gradient import BackwardCheckpointed, Direct, Forward
from dynamiqs.integrators.core.low_rank_integrator import _cholesky_qr2, expval_from_m
from dynamiqs.method import Kvaerno5, LinearSolver, LowRank, Tsit5

from ..integrator_tester import IntegratorTester
from ..order import TEST_INSTANT, TEST_LONG
from ..systems import dense_ocavity, dia_ocavity, mixed_ocavity, otdqubit

# we only test Tsit5 to keep the unit test suite fast, and Kvaerno5 for the real
# splitting used by implicit methods


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

    def test_correctness_implicit(self):
        method = LowRank(rank=4, ode_method=Kvaerno5(), key=jax.random.PRNGKey(0))
        self._test_correctness(dense_ocavity, method)

    def test_correctness_from_dm(self):
        # from a density matrix, the factor starts from its leading eigenvectors, and
        # from small random columns where its eigenvalues vanish (two are nonzero here)
        self._test_correctness(mixed_ocavity, _lowrank_method(mixed_ocavity))

    def test_gradient_from_dm(self):
        # through the eigendecomposition of rho0, whose other eigenvalues vanish
        method = _lowrank_method(mixed_ocavity)
        self._test_gradient(mixed_ocavity, method, BackwardCheckpointed())

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
        # the hook receives the current time and the low-rank factor m(t)
        result = system.run(method, save_extra=lambda t, m: (t, m))
        t, m = result.extra
        assert jnp.allclose(t, system.tsave)
        assert m.shape[-2:] == (system.n, rank)

    @pytest.mark.parametrize('system', [dense_ocavity])
    def test_save_extra_full_rank_by_default(self, system):
        method = LowRank(
            rank=system.n // 2, ode_method=Tsit5(), key=jax.random.PRNGKey(0)
        )
        # the hook receives the current time and the full-rank density matrix rho(t)
        result = system.run(method, save_extra=lambda t, rho: (t, rho))
        t, rho = result.extra
        assert jnp.allclose(t, system.tsave)
        assert rho.shape[-2:] == (system.n, system.n)


@pytest.mark.run(order=TEST_LONG)
def test_clipped_constant_jump_operator():
    # a constant operator clipped in time is not constant: the cavity decays only from
    # t = 0.5, <n>(t) = exp(-kappa max(t - 0.5, 0))
    n, kappa = 8, 1.0
    a = dq.destroy(n)
    tsave = jnp.linspace(0.0, 2.0, 9)
    result = dq.mesolve(
        dq.zeros(n),
        [dq.constant(jnp.sqrt(kappa) * a).clip(0.5, None)],
        dq.fock(n, 1),
        tsave,
        exp_ops=[a.dag() @ a],
        method=LowRank(rank=2, key=jax.random.PRNGKey(0)),
        progress_meter=False,
    )
    expected = jnp.exp(-kappa * jnp.maximum(tsave - 0.5, 0.0))
    assert jnp.allclose(result.expects[0].real, expected, atol=1e-4)


def test_expval_from_m_accepts_qarray():
    # expval_from_m must accept a QArray operator directly (e.g. dia layout)
    # without densifying it first
    n, rank = 4, 2
    op = dq.number(n, layout=dq.dia)
    m = jax.random.normal(jax.random.PRNGKey(0), (n, rank), dtype=jnp.complex128)

    expect = jnp.sum(jnp.conj(m) * (op.to_jax() @ m))
    assert jnp.allclose(expval_from_m(m, op), expect)


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('condition', [1e3, 1e12])
def test_cholesky_qr2(monkeypatch, backend, condition):
    # a = u s v, with b = a x0 + a residual orthogonal to the columns of a: the
    # least-squares solution is x0 while a is well conditioned, and Q R = a stays a
    # finite factorization when it is not (shifted Cholesky)
    monkeypatch.setattr(jax, 'default_backend', lambda: backend)
    n, rank = 300, 8
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    u, _ = jnp.linalg.qr(jax.random.normal(keys[0], (n, rank), dtype=jnp.complex128))
    v, _ = jnp.linalg.qr(jax.random.normal(keys[1], (rank, rank), dtype=jnp.complex128))
    a = u @ jnp.diag(jnp.logspace(0, -jnp.log10(condition), rank)) @ v
    q, r = _cholesky_qr2(a)
    assert jnp.allclose(q @ r, a, rtol=0, atol=1e-14)
    if condition < 1e7:
        assert jnp.allclose(q.conj().T @ q, jnp.eye(rank), rtol=0, atol=1e-14)
        x0 = jax.random.normal(keys[2], (rank, 2), dtype=jnp.complex128)
        w = jax.random.normal(keys[3], (n, 2), dtype=jnp.complex128)
        b = a @ x0 + 1e-3 * (w - u @ (u.conj().T @ w))
        x = jax.scipy.linalg.solve_triangular(r, q.conj().T @ b)
        assert jnp.allclose(x, x0, rtol=1e-8, atol=0)
