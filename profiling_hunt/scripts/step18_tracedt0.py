"""Step 18 - decisive test: bare diffrax, throw=False, but with t0/t1 extracted from
the TRACED ts array (t0=ts[0], t1=ts[-1]) instead of passed as Python float literals.

Every earlier bare-diffrax test passed t0=0.0, t1=0.1 as literals, which XLA
constant-folds (0.0 == 0.1 is statically False), dead-code-eliminating diffrax's
t0==t1 conditional entirely -- that conditional never appeared in the HLO. dynamiqs
computes t0/t1 as self.ts[0]/self.ts[-1], genuinely dynamic values the compiler cannot
fold, so the conditional (and its copy, found in Step 11's HLO trace) survives for
real. This isolates exactly that one variable.
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


def solve(nt, traced_t0t1):
    M = jax.random.normal(jax.random.key(0), (N, N), dtype=jnp.complex128)
    y0 = jax.random.normal(jax.random.key(1), (N, N), dtype=jnp.complex128)
    term = dx.ODETerm(lambda t, y, args: M @ y)
    ts = jnp.linspace(0.0, 0.1, nt)
    saveat = dx.SaveAt(ts=ts)

    if traced_t0t1:
        t0, t1 = ts[0], ts[-1]   # dynamic, traced -- exactly dynamiqs' self.t0/self.t1
    else:
        t0, t1 = 0.0, 0.1        # Python float literals -- compile-time constants

    def f(y0):
        return dx.diffeqsolve(
            term, dx.Tsit5(), t0=t0, t1=t1, dt0=None, y0=y0, saveat=saveat,
            stepsize_controller=dx.PIDController(rtol=1e-6, atol=1e-6),
            max_steps=100_000, throw=False,
        )

    return f, y0


ap = argparse.ArgumentParser()
ap.add_argument("--traced", type=int, required=True)
ap.add_argument("--ntsave", type=int, required=True)
ap.add_argument("--json", type=str, required=True)
args = ap.parse_args()

dev = jax.local_devices()[0]
f, y0 = solve(args.ntsave, bool(args.traced))
rec = {"traced": bool(args.traced), "ntsave": args.ntsave, "R": R}
try:
    out = jax.block_until_ready(f(y0))
    pk = dev.memory_stats()["peak_bytes_in_use"]
    print(f"### traced_t0t1={bool(args.traced)} ntsave={args.ntsave}  "
          f"peak={pk:,d} B = {pk/GIB:.5f} GiB = {pk/R:.4f} R", flush=True)
    rec.update({"status": "ok", "peak": pk, "peak_in_R": pk / R})
except Exception as exc:  # noqa: BLE001
    pk = dev.memory_stats()["peak_bytes_in_use"]
    print(f"### traced_t0t1={bool(args.traced)} ntsave={args.ntsave}: RAISED "
          f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:100]}",
          flush=True)
    rec.update({"status": "EXC", "peak": pk, "peak_in_R": pk / R})
finally:
    gc.collect()
    jax.clear_caches()

with open(args.json, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
