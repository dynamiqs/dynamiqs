"""Step 16 - the EXACT proposed dynamiqs patch (throw=False + scalar error_if inside
DiffraxIntegrator.run), swept over ntsave, ONE PROCESS PER (variant, ntsave) PAIR,
measured with device.memory_stats()["peak_bytes_in_use"] -- the real allocator, not
compiled.memory_analysis(). Slope and intercept both FIT from 5 points, not assumed.

This is the same patch as Step 10/11 (diffeqsolve(throw=False) + one error_if on
result.infos.nsteps), run through the real dq.mesolve() call, at N=1000 so the
trajectory clears the ~48 MiB fixed floor at every ntsave tested (Step 12's mistake
at N=64 is not repeated here).
"""

import argparse
import gc
import json
import warnings
from typing import cast

import jax

jax.config.update("jax_enable_x64", True)

import diffrax as dx  # noqa: E402
import dynamiqs as dq  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from diffrax._solution import is_okay  # noqa: E402
from dynamiqs.integrators.core.diffrax_integrator import DiffraxIntegrator  # noqa: E402

GIB = 1024**3
N = 1000
R = 16 * N * N
ALPHA, K2, K1, DELTA = 2.0, 1.0, 0.05, 0.2


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


ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True, choices=["baseline", "patched"])
ap.add_argument("--ntsave", type=int, required=True)
ap.add_argument("--json", type=str, required=True)
args = ap.parse_args()

if args.variant == "patched":
    DiffraxIntegrator.diffeqsolve = diffeqsolve_no_throw
    DiffraxIntegrator.run = run_patched

dev = jax.local_devices()[0]
nt = args.ntsave
a = dq.destroy(N)
H = DELTA * dq.dag(a) @ a
Ls = [np.sqrt(K2) * (a @ a - ALPHA**2 * dq.eye(N)), np.sqrt(K1) * a]
rho0 = dq.fock_dm(N, 0).asdense()
ts = jnp.linspace(0.0, 0.1, nt)
method = dq.method.Tsit5(rtol=1e-6, atol=1e-6, max_steps=100_000)

rec = {"variant": args.variant, "ntsave": nt, "R": R}
try:
    res = jax.block_until_ready(
        dq.mesolve(H, Ls, rho0, ts, method=method, save_states=True,
                   progress_meter=False)
    )
    pk = dev.memory_stats()["peak_bytes_in_use"]
    st = np.asarray(res.states)
    print(f"### {args.variant}  N={N} ntsave={nt}  "
          f"peak={pk:,d} B = {pk / GIB:.5f} GiB = {pk / R:.4f} R  "
          f"finite={np.isfinite(st).all()}", flush=True)
    rec.update({"status": "ok", "peak": pk, "peak_in_R": pk / R,
                "finite": bool(np.isfinite(st).all())})
except Exception as exc:  # noqa: BLE001
    pk = dev.memory_stats()["peak_bytes_in_use"]
    print(f"### {args.variant}  N={N} ntsave={nt}: RAISED {type(exc).__name__}: "
          f"{str(exc).strip().splitlines()[0][:100]}", flush=True)
    rec.update({"status": "EXC", "peak": pk, "peak_in_R": pk / R})
finally:
    gc.collect()
    jax.clear_caches()

with open(args.json, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
