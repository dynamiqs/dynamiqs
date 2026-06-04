import jax
import jax.numpy as jnp

import dynamiqs as dq
from dynamiqs.gradient import HigherOrder
from dynamiqs.method import Tsit5


def test_mesolve_higher_order_gradient_matches_closed_qubit_hessian():
    t_final = 0.4
    omega = 0.7
    tsave = jnp.linspace(0.0, t_final, 9)
    rho0 = dq.basis(2, 0).todm()
    sx = dq.sigmax()
    sz = dq.sigmaz()
    method = Tsit5(rtol=1e-8, atol=1e-8)

    def final_z_expectation(omega):
        result = dq.mesolve(
            0.5 * omega * sx,
            [],
            rho0,
            tsave,
            exp_ops=[sz],
            method=method,
            gradient=HigherOrder(),
            save_states=False,
            progress_meter=False,
        )
        return jnp.real(result.expects[0, -1])

    actual = jax.hessian(final_z_expectation)(omega)
    expected = -(t_final**2) * jnp.cos(omega * t_final)

    assert jnp.allclose(actual, expected, rtol=1e-4, atol=1e-5)
