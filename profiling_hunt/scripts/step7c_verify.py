"""Step 7c - does EQX_ON_ERROR=off actually deliver 1 R per saved state on the
allocator, does it change the answer, and what does it cost in safety?

Step 7b showed, in the compiler's static accounting, that the trajectory slope falls
3 R -> 2 R -> 1 R as EQX_ON_ERROR goes raise -> nan -> off. This checks the same thing
three more ways, at a realistic size:

  1. peak_bytes_in_use, the metric the whole hunt is built on, not the compiler's
     estimate.
  2. whether the returned states are bit identical to the default setting, so that
     the saving is not being bought with a different answer.
  3. what happens on a solve that genuinely fails (max_steps far too small): does the
     failure still surface, or does it go silent?
"""

import argparse
import gc
import json
import os

import jax

jax.config.update("jax_enable_x64", True)

import dynamiqs as dq  # noqa: E402
import equinox  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

GIB = 1024**3
ALPHA, K2, K1, DELTA, T_FINAL = 2.0, 1.0, 0.05, 0.2, 0.1

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=2000)
ap.add_argument("--ntsave", type=int, default=101)
ap.add_argument("--max-steps", type=int, default=100_000)
ap.add_argument("--json", type=str, required=True)
args = ap.parse_args()

dev = jax.local_devices()[0]
N, nt = args.N, args.ntsave
R = 16 * N * N
setting = os.environ.get("EQX_ON_ERROR", "<unset>")

a = dq.destroy(N)
H = DELTA * dq.dag(a) @ a
Ls = [np.sqrt(K2) * (a @ a - ALPHA**2 * dq.eye(N)), np.sqrt(K1) * a]
rho0 = dq.fock_dm(N, 0).asdense()
ts = jnp.linspace(0.0, T_FINAL, nt)

print("#" * 100, flush=True)
print(f"EQX_ON_ERROR={setting} (equinox sees {equinox._config.EQX_ON_ERROR})  "
      f"N={N} ntsave={nt} max_steps={args.max_steps}", flush=True)
print(f"R = {R:,d} B   trajectory = nt*R = {nt * R / GIB:.4f} GiB", flush=True)
print(f"published model, 3 R (nt+1) = {3 * R * (nt + 1) / GIB:.4f} GiB", flush=True)
print("#" * 100, flush=True)

rec = {"setting": equinox._config.EQX_ON_ERROR, "N": N, "ntsave": nt, "R": R,
       "max_steps": args.max_steps}
try:
    res = dq.mesolve(
        H, Ls, rho0, ts,
        method=dq.method.Tsit5(rtol=1e-6, atol=1e-6, max_steps=args.max_steps),
        save_states=True, progress_meter=False)
    jax.block_until_ready(res)
    st = dev.memory_stats()
    pk = st["peak_bytes_in_use"]
    states = np.asarray(res.states.to_jax() if hasattr(res.states, "to_jax")
                        else res.states)
    finite = bool(np.isfinite(states).all())
    # a cheap fingerprint of the answer, to compare across settings
    trace_last = complex(np.trace(states[-1]))
    checksum = float(np.abs(states).sum())
    print(f"  peak = {pk:,d} B = {pk / GIB:.4f} GiB = {pk / R:.3f} R", flush=True)
    print(f"  peak / (nt*R) = {pk / (nt * R):.4f}   "
          f"slope implied = {(pk - 3 * R) / (nt * R):.4f} R per saved state",
          flush=True)
    print(f"  states shape={states.shape}  all finite={finite}", flush=True)
    print(f"  trace(rho_final) = {trace_last:.12f}", flush=True)
    print(f"  checksum sum|rho| = {checksum:.10e}", flush=True)
    rec.update({"status": "ok", "peak": pk, "peak_in_R": pk / R,
                "peak_over_traj": pk / (nt * R), "finite": finite,
                "trace_final_re": trace_last.real, "trace_final_im": trace_last.imag,
                "checksum": checksum})
except Exception as exc:  # noqa: BLE001
    st = dev.memory_stats()
    print(f"  !! {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    rec.update({"status": "EXC", "exc": type(exc).__name__, "err": str(exc)[:300],
                "peak": st["peak_bytes_in_use"]})
finally:
    gc.collect()
    jax.clear_caches()

with open(args.json, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
