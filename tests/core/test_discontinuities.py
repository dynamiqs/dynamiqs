import diffrax as dx
import jax.numpy as jnp
import numpy as np

import dynamiqs as dq
from dynamiqs.integrators.core.diffrax_integrator import (
    MESolveDiffraxIntegrator,
    mesolve_tsit5_integrator_constructor,
)
from dynamiqs.options import Options
from dynamiqs.result import MESolveResult


def integrator(H: dq.TimeQArray, Ls: list[dq.TimeQArray]) -> MESolveDiffraxIntegrator:
    return mesolve_tsit5_integrator_constructor(
        ts=jnp.linspace(0.0, 1.0, 11),
        y0=dq.fock_dm(2, 0),
        method=dq.method.Tsit5(),
        gradient=None,
        result_class=MESolveResult,
        options=Options(),
        H=H,
        Ls=Ls,
        Es=[],
    )


def test_adaptive_steps_are_clipped_at_discontinuities():
    # `dq.pwc` values change instantaneously, and an adaptive step spanning such a jump
    # integrates a vector field that is wrong over part of the step, at an accuracy loss
    # its own error estimate cannot report. The steps must be clipped there.
    times = jnp.array([0.0, 0.3, 0.7, 1.0])
    H = dq.pwc(times, jnp.array([1.0, 0.0, 1.0]), dq.sigmax())
    Ls = [dq.pwc(jnp.array([0.0, 0.5]), jnp.array([0.2]), dq.sigmam())]

    controller = integrator(H, Ls).stepsize_controller
    assert isinstance(controller, dx.ClipStepSizeController)
    # every discontinuity of `H` and of the jump operators, in one sorted array
    np.testing.assert_allclose(
        controller.jump_ts, jnp.array([0.0, 0.0, 0.3, 0.5, 0.7, 1.0])
    )


def test_constant_operators_need_no_clipping():
    controller = integrator(dq.constant(dq.sigmax()), [dq.constant(dq.sigmam())])
    assert isinstance(controller.stepsize_controller, dx.PIDController)
