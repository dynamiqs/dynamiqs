import diffrax as dx
import jax.numpy as jnp
import pytest

from dynamiqs.integrators.core.diffrax_integrator import pid_controller
from dynamiqs.method import Kvaerno5

from ..order import TEST_SHORT

diffrax_nan_step = pytest.mark.xfail(
    reason='diffrax#774: a failed implicit step with a complex `y` gives a NaN step '
    'size; an XPASS means the installed Diffrax fixed it (e.g. diffrax#775): raise '
    'the `diffrax` lower bound to that release, then remove `error_norm`',
    raises=AssertionError,
    strict=True,
)


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize(
    'controller',
    [
        pytest.param(dx.PIDController(rtol=1e-3, atol=1e-3), marks=diffrax_nan_step),
        pid_controller(Kvaerno5(rtol=1e-3, atol=1e-3)),
    ],
    ids=['diffrax', 'dynamiqs'],
)
def test_complex_failed_step_is_rejected(controller):
    # `dt0=1` is too large for the root find, so the first step fails and reports
    # `y_error=inf`: it must be rejected and retried with a smaller step, not give a
    # NaN step size until `max_steps`
    sol = dx.diffeqsolve(
        dx.ODETerm(lambda t, y, args: -10 * y**3),  # noqa: ARG005
        dx.Kvaerno5(),
        0.0,
        1.0,
        1.0,
        jnp.array(1.0 + 0j),
        stepsize_controller=controller,
        max_steps=1000,
        throw=False,
    )
    assert sol.result == dx.RESULTS.successful
    assert sol.stats['num_rejected_steps'] > 0
    # y(t) = 1 / sqrt(1 + 20 t)
    assert jnp.allclose(sol.ys[-1], 21**-0.5, rtol=1e-2)
