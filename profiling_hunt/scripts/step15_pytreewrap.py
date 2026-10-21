"""Step 15 - does wrapping the state in a trivial pytree node (an eqx.Module with one
array field, mimicking QArray's shape as a pytree) reproduce the extra 1 R, even
with a bare M@y vector field, one SubSaveAt, throw=False?

Every previous bare-diffrax test carried a RAW array as `y`. dynamiqs carries a QArray
(an eqx.Module wrapping one array field + static dims metadata) as `y` all the way
through the loop, in and out of `buffer_at_set`. This isolates that one variable.
"""
import argparse
import gc
import json

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import diffrax as dx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.tree_util as jtu  # noqa: E402
import numpy as np  # noqa: E402

GIB = 1024**3
N = 1000
R = 16 * N * N


class Wrapped(eqx.Module):
    """Minimal one-array-field pytree node, structurally like a QArray."""

    data: jax.Array
    dims: tuple = eqx.field(static=True)

    def __add__(self, other):
        return Wrapped(self.data + (other.data if isinstance(other, Wrapped) else other),
                       self.dims)

    def __mul__(self, s):
        return Wrapped(self.data * s, self.dims)

    __rmul__ = __mul__


def solve(nt, wrap_state):
    M = jax.random.normal(jax.random.key(0), (N, N), dtype=jnp.complex128)
    y0 = jax.random.normal(jax.random.key(1), (N, N), dtype=jnp.complex128)
    ts = jnp.linspace(0.0, 0.1, nt)

    if wrap_state:
        y0w = Wrapped(y0, (N, N))

        def vf(t, y, args):  # noqa: ARG001
            return Wrapped(M @ y.data, y.dims)

        def save_fn(t, y, args):  # noqa: ARG001
            return y  # the Wrapped pytree, exactly like SolveSaveMixin.save()

        saveat = dx.SaveAt(ts=ts, fn=save_fn)
        y0_arg = y0w
    else:
        def vf(t, y, args):  # noqa: ARG001
            return M @ y

        saveat = dx.SaveAt(ts=ts)
        y0_arg = y0

    term = dx.ODETerm(vf)

    def f(y0_):
        return dx.diffeqsolve(
            term, dx.Tsit5(), t0=0.0, t1=0.1, dt0=None, y0=y0_, saveat=saveat,
            stepsize_controller=dx.PIDController(rtol=1e-6, atol=1e-6),
            max_steps=100_000, throw=False,
        )

    return f, y0_arg


ap = argparse.ArgumentParser()
ap.add_argument("--wrap", type=int, required=True)
ap.add_argument("--ntsave", type=int, required=True)
ap.add_argument("--json", type=str, required=True)
args = ap.parse_args()

dev = jax.local_devices()[0]
f, y0 = solve(args.ntsave, bool(args.wrap))
rec = {"wrap": bool(args.wrap), "ntsave": args.ntsave, "R": R}
try:
    out = jax.block_until_ready(f(y0))
    pk = dev.memory_stats()["peak_bytes_in_use"]
    print(f"### wrap={bool(args.wrap)} ntsave={args.ntsave}  "
          f"peak={pk:,d} B = {pk/GIB:.5f} GiB = {pk/R:.4f} R", flush=True)
    rec.update({"status": "ok", "peak": pk, "peak_in_R": pk / R})
except Exception as exc:  # noqa: BLE001
    pk = dev.memory_stats()["peak_bytes_in_use"]
    print(f"### wrap={bool(args.wrap)} ntsave={args.ntsave}: RAISED "
          f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:120]}",
          flush=True)
    rec.update({"status": "EXC", "peak": pk, "peak_in_R": pk / R})
finally:
    gc.collect()
    jax.clear_caches()

with open(args.json, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
