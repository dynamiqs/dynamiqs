"""Step 8c - the small leaf fix at production size, on the real allocator."""
import argparse, gc, os
import jax
jax.config.update("jax_enable_x64", True)
import dynamiqs as dq, equinox as eqx, jax.numpy as jnp, jax.tree_util as jtu, numpy as np
from equinox._enum import EnumerationItem
from equinox._errors import branched_error_if

GIB = 1024**3
THRESH = 4096
_ORIG = EnumerationItem.error_if

def v4_small(self, token, pred):
    leaves, treedef = jtu.tree_flatten(token)
    idx = [i for i, x in enumerate(leaves) if eqx.is_array(x) and x.nbytes <= THRESH]
    if not idx:
        return _ORIG(self, token, pred)
    checked = branched_error_if([leaves[i] for i in idx], pred, self._value,
                                self._enumeration._index_to_message)
    out = list(leaves)
    for i, v in zip(idx, checked):
        out[i] = v
    return jtu.tree_unflatten(treedef, out)

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=2000)
ap.add_argument("--ntsave", type=int, default=101)
ap.add_argument("--max-steps", type=int, default=100_000)
ap.add_argument("--patch", type=int, default=0)
args = ap.parse_args()
if args.patch:
    EnumerationItem.error_if = v4_small

dev = jax.local_devices()[0]
N, nt = args.N, args.ntsave
R = 16 * N * N
a = dq.destroy(N); H = 0.2 * dq.dag(a) @ a
Ls = [1.0 * (a @ a - 4.0 * dq.eye(N)), np.sqrt(0.05) * a]
rho0 = dq.fock_dm(N, 0).asdense()
ts = jnp.linspace(0.0, 0.1, nt)
print(f"patch={'v4_small' if args.patch else 'BASELINE'}  N={N} ntsave={nt} "
      f"max_steps={args.max_steps}  traj = nt*R = {nt * R / GIB:.4f} GiB", flush=True)
try:
    res = dq.mesolve(H, Ls, rho0, ts,
                     method=dq.method.Tsit5(rtol=1e-6, atol=1e-6,
                                            max_steps=args.max_steps),
                     save_states=True, progress_meter=False)
    jax.block_until_ready(res)
    pk = dev.memory_stats()["peak_bytes_in_use"]
    st = np.asarray(res.states)
    print(f"  peak = {pk:,d} B = {pk / GIB:.4f} GiB = {pk / R:.3f} R", flush=True)
    print(f"  slope = {(pk - 3 * R) / (nt * R):.4f} R per saved state", flush=True)
    print(f"  finite={np.isfinite(st).all()}  trace={complex(np.trace(st[-1])):.12f}  "
          f"checksum={float(np.abs(st).sum()):.10e}", flush=True)
except Exception as exc:
    print(f"  RAISED {type(exc).__name__}: {str(exc).strip().splitlines()[0][:90]}",
          flush=True)
finally:
    gc.collect(); jax.clear_caches()
