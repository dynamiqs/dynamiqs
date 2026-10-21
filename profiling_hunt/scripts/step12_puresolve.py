"""Step 12 - pure diffrax, SaveAt(ts=...), throw True vs False, measured on the real
allocator with device.memory_stats()["peak_bytes_in_use"] -- not compiled.memory_analysis().

ONE (which, ntsave) PAIR PER PROCESS. peak_bytes_in_use is a high-water mark that never
resets within a process; looping over ntsave in a shared process (a mistake caught and
reverted before this ran) would let the largest ntsave's peak mask every smaller one.

Two questions:
  1. Does dq.mesolve's shipped baseline reproduce ~3.0 R under memory_stats() at N=64,
     matching the compile-time number, not just at N=2000 where it was checked before?
  2. In BARE diffrax -- no dynamiqs, no vmap wrapper, no QArray, no options object --
     does throw=False alone reach 1 R, or does it also plateau at 2 R? This isolates
     whether the loop-carry-to-output copy found in Step 11 is a dynamiqs-side
     artefact or is inherent to diffrax/equinox's own buffer machinery.
"""

import argparse
import gc
import json

import jax

jax.config.update("jax_enable_x64", True)

import diffrax as dx  # noqa: E402
import dynamiqs as dq  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

GIB = 1024**3
N = 1000
R = 16 * N * N  # complex128, matches the dynamiqs test size exactly


def bare_diffrax(nt, throw):
    M = jax.random.normal(jax.random.key(0), (N, N), dtype=jnp.complex128)
    y0 = jax.random.normal(jax.random.key(1), (N, N), dtype=jnp.complex128)
    term = dx.ODETerm(lambda t, y, args: M @ y)
    ts = jnp.linspace(0.0, 0.1, nt)
    saveat = dx.SaveAt(ts=ts)

    def f(y0):
        return dx.diffeqsolve(
            term, dx.Tsit5(), t0=0.0, t1=0.1, dt0=None, y0=y0, saveat=saveat,
            stepsize_controller=dx.PIDController(rtol=1e-6, atol=1e-6),
            max_steps=100_000, throw=throw,
        )

    return f, y0


def dynamiqs_solve(nt):
    ALPHA, K2, K1, DELTA = 2.0, 1.0, 0.05, 0.2
    a = dq.destroy(N)
    H = DELTA * dq.dag(a) @ a
    Ls = [np.sqrt(K2) * (a @ a - ALPHA**2 * dq.eye(N)), np.sqrt(K1) * a]
    rho0 = dq.fock_dm(N, 0).asdense()
    ts = jnp.linspace(0.0, 0.1, nt)
    method = dq.method.Tsit5(rtol=1e-6, atol=1e-6, max_steps=100_000)

    def f(rho0):
        return dq.mesolve(H, Ls, rho0, ts, method=method, save_states=True,
                          progress_meter=False)

    return f, rho0


ap = argparse.ArgumentParser()
ap.add_argument("--which", required=True,
                choices=["dynamiqs_baseline", "bare_throw_true", "bare_throw_false"])
ap.add_argument("--ntsave", type=int, required=True)
ap.add_argument("--json", type=str, required=True)
args = ap.parse_args()

dev = jax.local_devices()[0]
if args.which == "dynamiqs_baseline":
    f, y0 = dynamiqs_solve(args.ntsave)
elif args.which == "bare_throw_true":
    f, y0 = bare_diffrax(args.ntsave, True)
else:
    f, y0 = bare_diffrax(args.ntsave, False)

rec = {"which": args.which, "ntsave": args.ntsave, "R": R}
try:
    out = jax.block_until_ready(f(y0))
    pk = dev.memory_stats()["peak_bytes_in_use"]
    print(f"### {args.which}  N={N}  ntsave={args.ntsave}  R={R:,d} B", flush=True)
    print(f"  peak = {pk:,d} B = {pk / GIB:.6f} GiB = {pk / R:.4f} R", flush=True)
    rec.update({"status": "ok", "peak": pk, "peak_in_R": pk / R})
except Exception as exc:  # noqa: BLE001
    pk = dev.memory_stats()["peak_bytes_in_use"]
    print(f"### {args.which}  N={N}  ntsave={args.ntsave}: RAISED "
          f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:100]}",
          flush=True)
    rec.update({"status": "EXC", "exc": type(exc).__name__, "peak": pk,
                "peak_in_R": pk / R})
finally:
    gc.collect()
    jax.clear_caches()

with open(args.json, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
