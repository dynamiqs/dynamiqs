import diffrax as dx
import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.gradient import Direct, HigherOrder
from dynamiqs.method import Dopri5, Expm, Tsit5

from .order import TEST_SHORT

# Analytical reference for all tests below. For the qubit Hamiltonian
# H(w) = (w / 2) sigma_x with initial state |0>, the expectation value of
# sigma_z after a time t is
#     <sigma_z>(w) = cos(w t),
# so its second derivative with respect to w is
#     d2/dw2 <sigma_z>(w) = -t^2 cos(w t).
_T = 0.7
_W = 0.5
_TSAVE = jnp.array([0.0, _T])


# Hessian computation matches the analytical result much more tightly in double
# precision; keep the precision change local to this module to avoid leakage.
@pytest.fixture(scope='module', autouse=True)
def _double_precision():
    prev_x64 = jax.config.read('jax_enable_x64')
    dq.set_precision('double')
    yield
    dq.set_precision('double' if prev_x64 else 'single')


def _sesolve_sz(omega, gradient, method=Tsit5()):  # noqa: B008
    h = 0.5 * omega * dq.sigmax()
    psi0 = dq.basis(2, 0)
    res = dq.sesolve(
        h, psi0, _TSAVE, exp_ops=[dq.sigmaz()], gradient=gradient, method=method
    )
    return res.expects[0, -1].real


def _mesolve_sz(omega, gradient, method=Tsit5()):  # noqa: B008
    h = 0.5 * omega * dq.sigmax()
    rho0 = dq.todm(dq.basis(2, 0))
    res = dq.mesolve(
        h, [], rho0, _TSAVE, exp_ops=[dq.sigmaz()], gradient=gradient, method=method
    )
    return res.expects[0, -1].real


@pytest.mark.run(order=TEST_SHORT)
class TestHessian:
    @pytest.mark.parametrize('method', [Tsit5(), Dopri5()])
    def test_sesolve_scalar_hessian(self, method):
        f = lambda w: _sesolve_sz(w, HigherOrder(), method)
        hess = jax.hessian(f)(_W)
        expected = -(_T**2) * jnp.cos(_W * _T)
        assert jnp.allclose(hess, expected, atol=1e-5)

    def test_sesolve_hessian_matrix(self):
        # Two-parameter Hamiltonian H = ((a + b) / 2) sigma_x, so the full 2x2
        # Hessian is known analytically: every entry equals -t^2 cos((a + b) t).
        def f(p):
            a, b = p
            h = 0.5 * (a + b) * dq.sigmax()
            psi0 = dq.basis(2, 0)
            res = dq.sesolve(
                h, psi0, _TSAVE, exp_ops=[dq.sigmaz()], gradient=HigherOrder()
            )
            return res.expects[0, -1].real

        params = jnp.array([0.3, 0.2])  # a + b = _W = 0.5
        hess = jax.hessian(f)(params)
        expected = -(_T**2) * jnp.cos(_W * _T) * jnp.ones((2, 2))
        assert jnp.allclose(hess, expected, atol=1e-5)

    def test_mesolve_scalar_hessian(self):
        f = lambda w: _mesolve_sz(w, HigherOrder())
        hess = jax.hessian(f)(_W)
        expected = -(_T**2) * jnp.cos(_W * _T)
        assert jnp.allclose(hess, expected, atol=1e-5)

    def test_first_order_gradient_preserved(self):
        # HigherOrder must keep first-order gradients correct.
        f = lambda w: _sesolve_sz(w, HigherOrder())
        grad = jax.grad(f)(_W)
        expected = -_T * jnp.sin(_W * _T)
        assert jnp.allclose(grad, expected, atol=1e-5)

    def test_default_gradient_cannot_compute_hessian(self):
        # The default differentiation path (RecursiveCheckpointAdjoint) is not
        # compatible with jax.hessian; this is the limitation HigherOrder fixes.
        f = lambda w: _sesolve_sz(w, None)
        with pytest.raises(TypeError, match='custom_vjp'):
            jax.hessian(f)(_W)

    def test_supported_methods(self):
        # HigherOrder is enabled for the Diffrax ODE methods...
        assert Tsit5().supports_gradient(HigherOrder())
        assert Dopri5().supports_gradient(HigherOrder())
        # ...but not for methods using a different integrator (e.g. Expm).
        assert not Expm().supports_gradient(HigherOrder())

    def test_higher_order_rebuilds_solver_with_bounded_scan_kind(self, monkeypatch):
        # The behaviour that makes HigherOrder work is rebuilding the Runge-Kutta
        # method with scan_kind="bounded". Capture the solver actually handed to
        # diffrax to confirm HigherOrder sets it while Direct leaves it as None,
        # so the rebuild is genuinely exercised rather than dead code.
        captured = {}
        original = dx.diffeqsolve

        def spy(terms, solver, **kwargs):
            captured['scan_kind'] = solver.scan_kind
            return original(terms, solver, **kwargs)

        monkeypatch.setattr(dx, 'diffeqsolve', spy)

        jax.clear_caches()
        _sesolve_sz(_W, HigherOrder())
        assert captured['scan_kind'] == 'bounded'

        jax.clear_caches()
        _sesolve_sz(_W, Direct())
        assert captured['scan_kind'] is None
