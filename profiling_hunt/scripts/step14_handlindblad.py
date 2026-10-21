"""Step 14 - does a hand-written Lindblad RHS (raw jnp arrays, no dynamiqs, no
QArray) reproduce the extra 1 R, or does bare diffrax always stay at exactly 1 R
regardless of vector-field complexity?

If this reaches ~2 R, the missing aliasing is an XLA scheduling/heuristic issue that
shows up once the surrounding computation graph is big enough (more competing
temporaries around the save-buffer's finalization), independent of dynamiqs. If it
stays at ~1 R, the extra copy really is specific to something in dynamiqs' own
integrator/QArray machinery.
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
ALPHA, K2, K1, DELTA = 2.0, 1.0, 0.05, 0.2


def solve(nt, throw):
    key = jax.random.key(0)
    H = DELTA * jax.random.normal(key, (N, N), dtype=jnp.complex128)
    H = H + H.conj().T
    L1 = np.sqrt(K1) * jax.random.normal(jax.random.key(2), (N, N), dtype=jnp.complex128)
    L2 = np.sqrt(K2) * jax.random.normal(jax.random.key(3), (N, N), dtype=jnp.complex128)
    L1d, L2d = L1.conj().T, L2.conj().T
    L1dL1, L2dL2 = L1d @ L1, L2d @ L2
    rho0 = jax.random.normal(jax.random.key(1), (N, N), dtype=jnp.complex128)

    def vf(t, rho, args):  # noqa: ARG001
        out = -1j * (H @ rho - rho @ H)
        for L, Ld, LdL in ((L1, L1d, L1dL1), (L2, L2d, L2dL2)):
            out = out + L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)
        return out

    term = dx.ODETerm(vf)
    ts = jnp.linspace(0.0, 0.1, nt)
    saveat = dx.SaveAt(ts=ts)

    def f(rho0):
        return dx.diffeqsolve(
            term, dx.Tsit5(), t0=0.0, t1=0.1, dt0=None, y0=rho0, saveat=saveat,
            stepsize_controller=dx.PIDController(rtol=1e-6, atol=1e-6),
            max_steps=100_000, throw=throw,
        )

    return f, rho0


ap = argparse.ArgumentParser()
ap.add_argument("--throw", type=int, required=True)
ap.add_argument("--ntsave", type=int, required=True)
ap.add_argument("--json", type=str, required=True)
args = ap.parse_args()

dev = jax.local_devices()[0]
f, rho0 = solve(args.ntsave, bool(args.throw))
rec = {"throw": bool(args.throw), "ntsave": args.ntsave, "R": R}
try:
    out = jax.block_until_ready(f(rho0))
    pk = dev.memory_stats()["peak_bytes_in_use"]
    print(f"### throw={bool(args.throw)} ntsave={args.ntsave}  "
          f"peak={pk:,d} B = {pk/GIB:.5f} GiB = {pk/R:.4f} R", flush=True)
    rec.update({"status": "ok", "peak": pk, "peak_in_R": pk / R})
except Exception as exc:  # noqa: BLE001
    pk = dev.memory_stats()["peak_bytes_in_use"]
    print(f"### throw={bool(args.throw)} ntsave={args.ntsave}: RAISED "
          f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:100]}",
          flush=True)
    rec.update({"status": "EXC", "peak": pk, "peak_in_R": pk / R})
finally:
    gc.collect()
    jax.clear_caches()

with open(args.json, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
