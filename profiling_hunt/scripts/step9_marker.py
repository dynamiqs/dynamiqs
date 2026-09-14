"""Step 9 - prototype the proposed design: no throw in diffrax, one scalar check in mesolve.

The proposal: stop letting diffrax pipe the Solution through `error_if`, carry only a
small "something went wrong" marker, and run a single `error_if` on that marker after
the solve completes.

Variants, all at N=2000, measured with peak_bytes_in_use:

  A_baseline      dq.mesolve as it ships
  B_marker        diffrax's error_if neutered (i.e. throw=False), then one error_if in
                  Python on a scalar marker `~isfinite(states).all()`, threaded back on
                  a small leaf of the Result so it cannot be eliminated
  C_marker_jit    B, with the whole call inside one jax.jit
  D_baseline_jit  A, with the whole call inside one jax.jit

C and D exist to locate the copy that remains after the diffrax cost is removed: the
compiler says 1 R is reachable but a direct dq.mesolve call delivers 2 R, which points
at dynamiqs postprocessing running outside its own inner jit.

Each variant is also run against max_steps=1, which must still be reported.
"""

import argparse
import gc

import jax

jax.config.update("jax_enable_x64", True)

import dynamiqs as dq  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from equinox._enum import EnumerationItem  # noqa: E402

GIB = 1024**3
_ORIG = EnumerationItem.error_if


def no_throw(self, token, pred):
    """Stand in for diffeqsolve(..., throw=False): the payload is untouched."""
    return token


ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True,
                choices=["A_baseline", "B_marker", "C_marker_jit", "D_baseline_jit"])
ap.add_argument("--N", type=int, default=2000)
ap.add_argument("--ntsave", type=int, default=101)
ap.add_argument("--max-steps", type=int, default=100_000)
args = ap.parse_args()

marker_variant = args.variant in ("B_marker", "C_marker_jit")
use_jit = args.variant in ("C_marker_jit", "D_baseline_jit")
if marker_variant:
    EnumerationItem.error_if = no_throw

dev = jax.local_devices()[0]
N, nt = args.N, args.ntsave
R = 16 * N * N
a = dq.destroy(N)
H = 0.2 * dq.dag(a) @ a
Ls = [1.0 * (a @ a - 4.0 * dq.eye(N)), np.sqrt(0.05) * a]
rho0 = dq.fock_dm(N, 0).asdense()
ts = jnp.linspace(0.0, 0.1, nt)
method = dq.method.Tsit5(rtol=1e-6, atol=1e-6, max_steps=args.max_steps)


def solve(r):
    res = dq.mesolve(H, Ls, r, ts, method=method, save_states=True,
                     progress_meter=False)
    if marker_variant:
        # the marker: one scalar reduction over the trajectory, no copy of it
        bad = jnp.invert(jnp.isfinite(res.states.to_jax()).all())
        # thread the check on a small leaf so it survives dead code elimination
        checked = eqx.error_if(res.infos.nsteps, bad, "dynamiqs: the solve failed.")
        res = eqx.tree_at(lambda x: x.infos.nsteps, res, checked)
    return res


fn = jax.jit(solve) if use_jit else solve
print(f"### {args.variant}  N={N} ntsave={nt} max_steps={args.max_steps}  "
      f"traj = {nt * R / GIB:.4f} GiB", flush=True)
try:
    res = jax.block_until_ready(fn(rho0))
    pk = dev.memory_stats()["peak_bytes_in_use"]
    st = np.asarray(res.states)
    print(f"  peak = {pk:,d} B = {pk / GIB:.4f} GiB = {pk / R:.3f} R", flush=True)
    print(f"  slope = {(pk - 3 * R) / (nt * R):.4f} R per saved state", flush=True)
    print(f"  finite={np.isfinite(st).all()}  "
          f"trace={complex(np.trace(st[-1])):.12f}  "
          f"checksum={float(np.abs(st).sum()):.10e}", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"  RAISED {type(exc).__name__}: "
          f"{str(exc).strip().splitlines()[0][:90]}", flush=True)
finally:
    gc.collect()
    jax.clear_caches()
