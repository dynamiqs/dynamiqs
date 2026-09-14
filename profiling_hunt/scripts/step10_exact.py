"""Step 10 - test the EXACT proposed patch, for peak memory AND for run time.

Step 9 tested a stand in: it neutered `EnumerationItem.error_if` globally and ran the
check in a wrapper outside dynamiqs, using an `isfinite` reduction as the marker. This
tests the thing actually proposed: `throw=False` on the real diffeqsolve call, and

    failed = jnp.invert(is_okay(solution.result))
    result = eqx.tree_at(lambda r: r.infos.nsteps, result,
                         solution.result.error_if(result.infos.nsteps, failed))

placed inside DiffraxIntegrator.run(), patched at class level so nothing on disk is
touched.

Variants:
  baseline      dynamiqs as it ships
  result_code   the exact patch above. Marker is the diffrax result enum: a scalar,
                already computed, so it costs no extra reads.
  plus_nan      as result_code, but OR in ~isfinite(ysave).all(), which catches a
                solver that reports success and returns garbage. This reads the whole
                trajectory, so it is the variant whose run time cost is in question.

Run time is measured after a warm up call, so compile time is excluded and reported
separately. Peak memory is read once, on a fresh process, before any repeat runs.
"""

import argparse
import gc
import time
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
_ORIG_DIFFEQSOLVE = DiffraxIntegrator.diffeqsolve
_ORIG_RUN = DiffraxIntegrator.run


def diffeqsolve_no_throw(self, t0, t1, y0, saveat, event=None, dtmax=None):
    """As DiffraxIntegrator.diffeqsolve, but with throw=False."""
    import warnings

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
            throw=False,  # <-- the change: stop piping the Solution through error_if
            progress_meter=self.options.progress_meter.to_diffrax(),
        )


def make_run(check_nan: bool):
    def run(self):
        fn = lambda t, y, args: self.save(y)  # noqa: ARG005
        subsaveat_a = dx.SubSaveAt(ts=self.ts, fn=fn)
        subsaveat_b = dx.SubSaveAt(t1=True)
        saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b])

        solution = self.diffeqsolve(self.t0, self.t1, self.y0, saveat)

        ys = cast(tuple, solution.ys)
        saved = self.postprocess_saved(*ys)
        result = self.result(saved, infos=self.infos(solution.stats))

        # the proposed check: token is one scalar, the trajectory never enters it
        failed = jnp.invert(is_okay(solution.result))
        if check_nan:
            leaves = [x for x in jax.tree.leaves(saved.ysave) if eqx.is_array(x)]
            for leaf in leaves:
                failed = failed | jnp.invert(jnp.isfinite(leaf).all())
        result = eqx.tree_at(
            lambda r: r.infos.nsteps, result,
            solution.result.error_if(result.infos.nsteps, failed),
        )
        return result

    return run


ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True,
                choices=["baseline", "result_code", "plus_nan"])
ap.add_argument("--N", type=int, default=2000)
ap.add_argument("--ntsave", type=int, default=101)
ap.add_argument("--max-steps", type=int, default=100_000)
ap.add_argument("--repeats", type=int, default=5)
args = ap.parse_args()

if args.variant != "baseline":
    DiffraxIntegrator.diffeqsolve = diffeqsolve_no_throw
    DiffraxIntegrator.run = make_run(check_nan=(args.variant == "plus_nan"))

dev = jax.local_devices()[0]
N, nt = args.N, args.ntsave
R = 16 * N * N
a = dq.destroy(N)
H = 0.2 * dq.dag(a) @ a
Ls = [1.0 * (a @ a - 4.0 * dq.eye(N)), np.sqrt(0.05) * a]
rho0 = dq.fock_dm(N, 0).asdense()
ts = jnp.linspace(0.0, 0.1, nt)
method = dq.method.Tsit5(rtol=1e-6, atol=1e-6, max_steps=args.max_steps)

solve = lambda r: dq.mesolve(H, Ls, r, ts, method=method, save_states=True,
                             progress_meter=False)

print(f"### {args.variant}  N={N} ntsave={nt} max_steps={args.max_steps}  "
      f"traj = {nt * R / GIB:.4f} GiB", flush=True)
try:
    t0 = time.perf_counter()
    res = jax.block_until_ready(solve(rho0))
    t_first = time.perf_counter() - t0
    pk = dev.memory_stats()["peak_bytes_in_use"]
    st = np.asarray(res.states)

    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(solve(rho0))
        times.append(time.perf_counter() - t0)
    times = np.array(times)

    print(f"  peak      = {pk:,d} B = {pk / GIB:.4f} GiB = {pk / R:.3f} R", flush=True)
    print(f"  slope     = {(pk - 3 * R) / (nt * R):.4f} R per saved state", flush=True)
    print(f"  first call (incl. compile) = {t_first:.3f} s", flush=True)
    print(f"  warm runs  = {times.mean():.4f} s +- {times.std():.4f}  "
          f"(min {times.min():.4f}, n={len(times)})", flush=True)
    print(f"  finite={np.isfinite(st).all()}  "
          f"trace={complex(np.trace(st[-1])):.12f}  "
          f"checksum={float(np.abs(st).sum()):.10e}", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"  RAISED {type(exc).__name__}: "
          f"{str(exc).strip().splitlines()[0][:100]}", flush=True)
finally:
    gc.collect()
    jax.clear_caches()
