"""Step 13 - is the extra 1 R caused by dynamiqs' TWO SubSaveAts (ts=... AND a
separate t1=True), rather than anything QArray or Result related?

dq.mesolve always builds:
    saveat = dx.SaveAt(subs=[dx.SubSaveAt(ts=self.ts, fn=fn), dx.SubSaveAt(t1=True)])
Step 12's bare diffrax test used a single dx.SaveAt(ts=ts) -- ONE SubSaveAt. This
reproduces dynamiqs' exact saveat shape in bare diffrax, still throw=False, still no
dynamiqs, no QArray, nothing else changed.
"""
import argparse
import gc
import json

import jax

jax.config.update("jax_enable_x64", True)

import diffrax as dx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

GIB = 1024**3
N = 1000
R = 16 * N * N


def solve(nt, two_subs):
    M = jax.random.normal(jax.random.key(0), (N, N), dtype=jnp.complex128)
    y0 = jax.random.normal(jax.random.key(1), (N, N), dtype=jnp.complex128)
    term = dx.ODETerm(lambda t, y, args: M @ y)
    ts = jnp.linspace(0.0, 0.1, nt)
    if two_subs:
        fn = lambda t, y, args: y  # noqa: ARG005
        saveat = dx.SaveAt(subs=[dx.SubSaveAt(ts=ts, fn=fn), dx.SubSaveAt(t1=True)])
    else:
        saveat = dx.SaveAt(ts=ts)

    def f(y0):
        return dx.diffeqsolve(
            term, dx.Tsit5(), t0=0.0, t1=0.1, dt0=None, y0=y0, saveat=saveat,
            stepsize_controller=dx.PIDController(rtol=1e-6, atol=1e-6),
            max_steps=100_000, throw=False,
        )

    return f, y0


ap = argparse.ArgumentParser()
ap.add_argument("--two-subs", type=int, required=True)
ap.add_argument("--ntsave", type=int, required=True)
ap.add_argument("--json", type=str, required=True)
args = ap.parse_args()

dev = jax.local_devices()[0]
f, y0 = solve(args.ntsave, bool(args.two_subs))
rec = {"two_subs": bool(args.two_subs), "ntsave": args.ntsave, "R": R}
try:
    out = jax.block_until_ready(f(y0))
    pk = dev.memory_stats()["peak_bytes_in_use"]
    print(f"### two_subs={bool(args.two_subs)} ntsave={args.ntsave}  "
          f"peak={pk:,d} B = {pk/GIB:.5f} GiB = {pk/R:.4f} R", flush=True)
    rec.update({"status": "ok", "peak": pk, "peak_in_R": pk / R})
except Exception as exc:  # noqa: BLE001
    pk = dev.memory_stats()["peak_bytes_in_use"]
    print(f"### two_subs={bool(args.two_subs)} ntsave={args.ntsave}: RAISED "
          f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:100]}",
          flush=True)
    rec.update({"status": "EXC", "peak": pk, "peak_in_R": pk / R})
finally:
    gc.collect()
    jax.clear_caches()

with open(args.json, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
