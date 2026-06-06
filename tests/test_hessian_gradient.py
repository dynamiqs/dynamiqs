import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.gradient import Hessian
from dynamiqs.method import Tsit5


@pytest.fixture(scope='module', autouse=True)
def _double_precision():
    prev_x64 = jax.config.read('jax_enable_x64')
    dq.set_precision('double')
    yield
    dq.set_precision('double' if prev_x64 else 'single')


def _qubit_hessian_problem(solve):
    sx = dq.sigmax()
    sz = dq.sigmaz()
    psi0 = dq.basis(2, 0)
    tsave = jnp.asarray([0.0, 0.7])
    method = Tsit5(rtol=1e-10, atol=1e-10)

    def final_sigmaz_expectation(omega):
        H = 0.5 * omega * sx
        result = solve(
            H,
            psi0,
            tsave,
            exp_ops=[sz],
            method=method,
            gradient=Hessian(),
            progress_meter=False,
        )
        return result.expects[0, -1].real

    omega = jnp.asarray(0.3)
    return jax.hessian(final_sigmaz_expectation)(omega), -(tsave[-1] ** 2) * jnp.cos(
        omega * tsave[-1]
    )


def test_sesolve_hessian_matches_analytical_qubit():
    hessian, expected = _qubit_hessian_problem(dq.sesolve)
    assert jnp.allclose(hessian, expected, rtol=1e-6, atol=1e-6)


def test_mesolve_hessian_matches_analytical_closed_qubit():
    def solve(H, psi0, tsave, *, exp_ops, method, gradient, progress_meter):
        return dq.mesolve(
            H,
            [],
            psi0.todm(),
            tsave,
            exp_ops=exp_ops,
            method=method,
            gradient=gradient,
            progress_meter=progress_meter,
        )

    hessian, expected = _qubit_hessian_problem(solve)
    assert jnp.allclose(hessian, expected, rtol=1e-6, atol=1e-6)
