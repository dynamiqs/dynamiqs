import diffrax as dx
import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.gradient import BackwardCheckpointed, Direct, HigherOrder
from dynamiqs.integrators.core.diffrax_integrator import hessian_compatible_solver
from dynamiqs.method import Dopri5, Dopri8, Euler, Kvaerno3, Kvaerno5, LowRank, Tsit5


def test_higher_order_gradient_uses_bounded_scan_kind():
    assert hessian_compatible_solver(dx.Tsit5(), Direct()).scan_kind is None
    assert hessian_compatible_solver(dx.Tsit5(), HigherOrder()).scan_kind == 'bounded'
    solver = hessian_compatible_solver(dx.Euler(), HigherOrder())
    assert not hasattr(solver, 'scan_kind')


@pytest.mark.parametrize(
    ('method', 'rtol', 'atol'),
    [
        pytest.param(Euler(dt=1e-3), 2e-3, 1e-4, id='Euler'),
        pytest.param(Dopri5(rtol=1e-7, atol=1e-7), 1e-5, 1e-5, id='Dopri5'),
        pytest.param(Dopri8(rtol=1e-8, atol=1e-8), 1e-5, 1e-5, id='Dopri8'),
        pytest.param(Tsit5(rtol=1e-7, atol=1e-7), 1e-5, 1e-5, id='Tsit5'),
    ],
)
def test_sesolve_hessian_matches_analytical_qubit_for_supported_methods(
    method, rtol, atol
):
    t = 0.3

    def loss(omega):
        result = dq.sesolve(
            0.5 * omega * dq.sigmax(),
            dq.fock(2, 0),
            jnp.array([0.0, t]),
            exp_ops=[dq.sigmaz()],
            method=method,
            gradient=HigherOrder(),
            progress_meter=False,
        )
        return result.expects[0, -1].real

    omega = 1.3
    expected = -(t**2) * jnp.cos(omega * t)

    assert jnp.allclose(jax.hessian(loss)(omega), expected, rtol=rtol, atol=atol)


def test_mesolve_hessian_matches_analytical_lossy_cavity():
    t = 0.3
    n = 8
    alpha0 = 0.5
    omega = 0.9

    def loss(kappa):
        a = dq.destroy(n)
        result = dq.mesolve(
            omega * a.dag() @ a,
            [jnp.sqrt(kappa) * a],
            dq.coherent(n, alpha0),
            jnp.array([0.0, t]),
            exp_ops=[dq.number(n)],
            method=Tsit5(rtol=1e-7, atol=1e-7),
            gradient=HigherOrder(),
            progress_meter=False,
        )
        return result.expects[0, -1].real

    kappa = 0.4
    expected = alpha0**2 * t**2 * jnp.exp(-kappa * t)

    assert jnp.allclose(jax.hessian(loss)(kappa), expected, rtol=1e-5, atol=1e-5)


def test_sesolve_hessian_matrix_two_parameters():
    # full 2x2 Hessian of <sigma_z>(t) for H = (wx/2) sigma_x + (wz/2) sigma_z
    t = 0.7

    def loss(params):
        wx, wz = params
        h = 0.5 * wx * dq.sigmax() + 0.5 * wz * dq.sigmaz()
        result = dq.sesolve(
            h,
            dq.fock(2, 0),
            jnp.array([0.0, t]),
            exp_ops=[dq.sigmaz()],
            method=Tsit5(rtol=1e-8, atol=1e-8),
            gradient=HigherOrder(),
            progress_meter=False,
        )
        return result.expects[0, -1].real

    # independent reference: H is time-independent, so the exact propagator is
    # U = expm(-i H t) and the reference Hessian comes from differentiating it directly
    sx = dq.sigmax().to_jax()
    sz = dq.sigmaz().to_jax()
    psi0 = dq.fock(2, 0).to_jax()

    def ref_loss(params):
        wx, wz = params
        h = 0.5 * wx * sx + 0.5 * wz * sz
        psi = jax.scipy.linalg.expm(-1j * h * t) @ psi0
        return (psi.conj().T @ sz @ psi).real.squeeze()

    params = jnp.array([1.3, 0.4])
    hessian = jax.hessian(loss)(params)
    reference = jax.hessian(ref_loss)(params)

    assert jnp.allclose(hessian, hessian.T, atol=1e-6)
    assert jnp.allclose(hessian, reference, rtol=1e-4, atol=1e-4)


def test_default_gradient_rejects_hessian():
    # regression guard: the default first-order path (RecursiveCheckpointAdjoint, a
    # custom_vjp) cannot be Hessian-differentiated; HigherOrder is what unlocks it.
    t = 0.7

    def loss(omega):
        result = dq.sesolve(
            0.5 * omega * dq.sigmax(),
            dq.fock(2, 0),
            jnp.array([0.0, t]),
            exp_ops=[dq.sigmaz()],
            method=Tsit5(rtol=1e-7, atol=1e-7),
            gradient=BackwardCheckpointed(),
            progress_meter=False,
        )
        return result.expects[0, -1].real

    # first-order gradient still works
    assert jnp.isfinite(jax.grad(loss)(1.3))
    # second-order differentiation is rejected
    with pytest.raises(TypeError):
        jax.hessian(loss)(1.3)


@pytest.mark.parametrize(
    'method',
    [
        pytest.param(Kvaerno3(rtol=1e-7, atol=1e-7), id='Kvaerno3'),
        pytest.param(Kvaerno5(rtol=1e-7, atol=1e-7), id='Kvaerno5'),
    ],
)
def test_implicit_solvers_reject_higher_order_gradient(method):
    with pytest.raises(
        ValueError,
        match=f'Method `{type(method).__name__}` does not support gradient'
        ' `HigherOrder`',
    ):
        dq.sesolve(
            dq.sigmax(),
            dq.fock(2, 0),
            jnp.array([0.0, 0.1]),
            exp_ops=[dq.sigmaz()],
            method=method,
            gradient=HigherOrder(),
            progress_meter=False,
        )


def test_lowrank_rejects_higher_order_gradient():
    with pytest.raises(
        ValueError, match='Method `LowRank` does not support gradient `HigherOrder`'
    ):
        dq.mesolve(
            dq.sigmax(),
            [],
            dq.fock_dm(2, 0),
            jnp.array([0.0, 0.1]),
            method=LowRank(rank=1, ode_method=Tsit5(), key=jax.random.PRNGKey(0)),
            gradient=HigherOrder(),
            progress_meter=False,
        )
