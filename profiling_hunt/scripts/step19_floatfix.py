"""Step 19 - apply float(tsave[0]), float(tsave[-1]) and rerun memory_stats.

Two variants, because WHERE the float() conversion happens matters:

  A) bare_float_fix: reproduces the fix at the only place it can legally run --
     BEFORE the jax.jit boundary, where `ts` is still a concrete (non-traced) array.
     This is mechanically identical to Step 18's "traced=False" case, but t0/t1 are
     *derived* via float(ts[0])/float(ts[-1]) rather than hand-typed literals, to
     confirm the derivation itself is equivalent to a literal once done pre-jit.

  B) dynamiqs_naive_property_fix: applies float(self.ts[0]) / float(self.ts[-1])
     literally inside DiffraxIntegrator.t0/.t1, exactly where the user's snippet
     would go if dropped in at the obvious call site. self.ts there is accessed
     from *inside* the already-jitted `_vectorized_mesolve` trace (jax.jit(f) in
     mesolve.py wraps everything up to and including DiffraxIntegrator.run()), so
     self.ts is a Tracer, not a concrete array -- this checks what actually happens.
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


# ---------- variant A: bare diffrax, float() applied pre-jit ----------
def bare_float_fix(nt):
    M = jax.random.normal(jax.random.key(0), (N, N), dtype=jnp.complex128)
    y0 = jax.random.normal(jax.random.key(1), (N, N), dtype=jnp.complex128)
    term = dx.ODETerm(lambda t, y, args: M @ y)
    ts = jnp.linspace(0.0, 0.1, nt)          # concrete, not inside any jit
    t0, t1 = float(ts[0]), float(ts[-1])     # the user's fix, applied pre-jit
    saveat = dx.SaveAt(ts=ts)

    def f(y0):
        return dx.diffeqsolve(
            term, dx.Tsit5(), t0=t0, t1=t1, dt0=None, y0=y0, saveat=saveat,
            stepsize_controller=dx.PIDController(rtol=1e-6, atol=1e-6),
            max_steps=100_000, throw=False,
        )

    return f, y0


# ---------- variant B: the naive fix dropped directly into DiffraxIntegrator ----------
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


def run_patched_naive_float(self):
    fn = lambda t, y, args: self.save(y)  # noqa: ARG005
    subsaveat_a = dx.SubSaveAt(ts=self.ts, fn=fn)
    subsaveat_b = dx.SubSaveAt(t1=True)
    saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b])
    t0 = float(self.ts[0])    # <-- the user's snippet, applied exactly here
    t1 = float(self.ts[-1])   # <-- self.ts is a Tracer at this point
    solution = self.diffeqsolve(t0, t1, self.y0, saveat)
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
ap.add_argument("--variant", required=True,
                choices=["bare_float_fix", "dynamiqs_naive_property_fix"])
ap.add_argument("--ntsave", type=int, required=True)
ap.add_argument("--json", type=str, required=True)
args = ap.parse_args()

dev = jax.local_devices()[0]
rec = {"variant": args.variant, "ntsave": args.ntsave, "R": R}

if args.variant == "bare_float_fix":
    f, y0 = bare_float_fix(args.ntsave)
    try:
        out = jax.block_until_ready(f(y0))
        pk = dev.memory_stats()["peak_bytes_in_use"]
        print(f"### {args.variant} ntsave={args.ntsave}  "
              f"peak={pk:,d} B = {pk/GIB:.5f} GiB = {pk/R:.4f} R", flush=True)
        rec.update({"status": "ok", "peak": pk, "peak_in_R": pk / R})
    except Exception as exc:  # noqa: BLE001
        pk = dev.memory_stats()["peak_bytes_in_use"]
        print(f"### {args.variant} ntsave={args.ntsave}: RAISED "
              f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:150]}",
              flush=True)
        rec.update({"status": "EXC", "peak": pk, "peak_in_R": pk / R})
    finally:
        gc.collect()
        jax.clear_caches()
else:
    DiffraxIntegrator.diffeqsolve = diffeqsolve_no_throw
    DiffraxIntegrator.run = run_patched_naive_float
    nt = args.ntsave
    a = dq.destroy(N)
    H = DELTA * dq.dag(a) @ a
    Ls = [np.sqrt(K2) * (a @ a - ALPHA**2 * dq.eye(N)), np.sqrt(K1) * a]
    rho0 = dq.fock_dm(N, 0).asdense()
    ts = jnp.linspace(0.0, 0.1, nt)
    method = dq.method.Tsit5(rtol=1e-6, atol=1e-6, max_steps=100_000)
    try:
        res = jax.block_until_ready(
            dq.mesolve(H, Ls, rho0, ts, method=method, save_states=True,
                       progress_meter=False)
        )
        pk = dev.memory_stats()["peak_bytes_in_use"]
        st = np.asarray(res.states)
        print(f"### {args.variant} ntsave={nt}  "
              f"peak={pk:,d} B = {pk/GIB:.5f} GiB = {pk/R:.4f} R  "
              f"finite={np.isfinite(st).all()}", flush=True)
        rec.update({"status": "ok", "peak": pk, "peak_in_R": pk / R})
    except Exception as exc:  # noqa: BLE001
        pk = dev.memory_stats()["peak_bytes_in_use"]
        print(f"### {args.variant} ntsave={nt}: RAISED {type(exc).__name__}: "
              f"{str(exc).strip().splitlines()[0][:200]}", flush=True)
        rec.update({"status": "EXC", "exc": type(exc).__name__, "peak": pk,
                    "peak_in_R": pk / R})
    finally:
        gc.collect()
        jax.clear_caches()

with open(args.json, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
