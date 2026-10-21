"""Step 11b - buffer-assignment dump for the throw=False + scalar error_if patch,
compiled at the SAME top-level boundary dq.mesolve itself uses (Step 11 showed this
predicts 2.0002 R, matching the real allocator's 2.045 R). Which instructions own
the remaining 1 R of temp, now that BOTH diffrax pure_callbacks are gone?
"""
import os
import sys

os.environ["XLA_FLAGS"] = f"--xla_dump_to={sys.argv[1]}"

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

import warnings  # noqa: E402
from typing import cast  # noqa: E402

import diffrax as dx  # noqa: E402
import dynamiqs as dq  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from diffrax._solution import is_okay  # noqa: E402
from dynamiqs.integrators._utils import astimeqarray  # noqa: E402
from dynamiqs.integrators.apis.mesolve import _vectorized_mesolve  # noqa: E402
from dynamiqs.integrators.core.diffrax_integrator import DiffraxIntegrator  # noqa: E402
from dynamiqs.options import Options  # noqa: E402
from dynamiqs.qarrays.utils import asqarray  # noqa: E402

N = 64
R = 16 * N * N
ALPHA, K2, K1, DELTA, T_FINAL = 2.0, 1.0, 0.05, 0.2, 0.1


def diffeqsolve_no_throw(self, t0, t1, y0, saveat, event=None, dtmax=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        stepsize_controller = self.stepsize_controller
        if dtmax is not None:
            get_dtmax = (
                (lambda c: c.controller.dtmax)
                if isinstance(stepsize_controller, dx.ClipStepSizeController)
                else (lambda c: c.dtmax)
            )
            stepsize_controller = eqx.tree_at(
                get_dtmax, stepsize_controller, dtmax, is_leaf=lambda x: x is None
            )
        return dx.diffeqsolve(
            self.terms, self.diffrax_solver, t0=t0, t1=t1, dt0=self.dt0, y0=y0,
            saveat=saveat, stepsize_controller=stepsize_controller,
            adjoint=self.adjoint, event=event, max_steps=self.max_steps,
            throw=False,
            progress_meter=self.options.progress_meter.to_diffrax(),
        )


def run_patched(self):
    fn = lambda t, y, args: self.save(y)  # noqa: ARG005
    subsaveat_a = dx.SubSaveAt(ts=self.ts, fn=fn)
    subsaveat_b = dx.SubSaveAt(t1=True)
    saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b])
    solution = self.diffeqsolve(self.t0, self.t1, self.y0, saveat)
    ys = cast(tuple, solution.ys)
    saved = self.postprocess_saved(*ys)
    result = self.result(saved, infos=self.infos(solution.stats))
    failed = jnp.invert(is_okay(solution.result))
    result = eqx.tree_at(
        lambda r: r.infos.nsteps, result,
        solution.result.error_if(result.infos.nsteps, failed),
    )
    return result


DiffraxIntegrator.diffeqsolve = diffeqsolve_no_throw
DiffraxIntegrator.run = run_patched

a = dq.destroy(N)
H = DELTA * dq.dag(a) @ a
Ls = [np.sqrt(K2) * (a @ a - ALPHA**2 * dq.eye(N)), np.sqrt(K1) * a]
rho0 = dq.fock_dm(N, 0).asdense()
H = astimeqarray(H)
Ls = [astimeqarray(L) for L in Ls]
rho0 = asqarray(rho0)
ts = jnp.linspace(0.0, T_FINAL, 40)
options = Options(save_states=True).initialise()
method = dq.method.Tsit5(rtol=1e-6, atol=1e-6, max_steps=100_000)

f = jax.jit(_vectorized_mesolve, static_argnames=('gradient', 'options'))
c = f.lower(H, Ls, rho0, ts, None, method, None, options).compile()
print(f"N={N} R={R:,d} B  ntsave=40  traj={40 * R:,d} B")
print(c.memory_analysis())
