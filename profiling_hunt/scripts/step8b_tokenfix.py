"""Step 8b - thread the error on the SMALL leaves of the Solution, not the trajectory.

Step 8 showed the two halves of the problem:
  * threading on a scalar marker costs 1.0019 R instead of 3.0002 R, so the saving is
    real and complete;
  * but a marker that is then thrown away is dead code, and the check disappears with
    it. Neither a discarded marker nor lax.optimization_barrier kept it alive.

The resolution is that the marker must remain in the returned PyTree. A jitted
function's outputs are never eliminated, so any leaf of the Solution that survives to
the caller will hold the callback live. Every leaf of a diffrax Solution is small
except `ys`. So: thread the error on all the small leaves and leave `ys` alone.

  v0_baseline   thread on the whole Solution                     (ships today)
  v3_one_cb     whole Solution, but drop the second TPU only callback
  v4_small      thread on every array leaf under 4 kB, i.e. ts, stats, result, and
                NOT ys
"""

import argparse

import jax

jax.config.update("jax_enable_x64", True)

import dynamiqs as dq  # noqa: E402
import equinox as eqx  # noqa: E402
import equinox._errors as E  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.tree_util as jtu  # noqa: E402
import numpy as np  # noqa: E402
from equinox._enum import EnumerationItem  # noqa: E402
from equinox._errors import branched_error_if  # noqa: E402

N = 64
R = 16 * N * N
THRESH = 4096  # bytes; separates ts/stats/result from ys at every size tested here
ALPHA, K2, K1, DELTA, T_FINAL = 2.0, 1.0, 0.05, 0.2, 0.1

_ORIGINAL_ERROR_IF = EnumerationItem.error_if
_ORIGINAL_ERROR = E._error


def v0_baseline(self, token, pred):
    return _ORIGINAL_ERROR_IF(self, token, pred)


def v4_small(self, token, pred):
    leaves, treedef = jtu.tree_flatten(token)
    idx = [i for i, x in enumerate(leaves) if eqx.is_array(x) and x.nbytes <= THRESH]
    if not idx:
        return _ORIGINAL_ERROR_IF(self, token, pred)
    checked = branched_error_if(
        [leaves[i] for i in idx], pred, self._value,
        self._enumeration._index_to_message)
    out = list(leaves)
    for i, v in zip(idx, checked):
        out[i] = v
    return jtu.tree_unflatten(treedef, out)


def _error_single_callback(x, pred, index, *, msgs, on_error, stack):
    """As equinox's `_error` for on_error='raise', minus the second, TPU only callback."""
    if on_error != "raise":
        return _ORIGINAL_ERROR(x, pred, index, msgs=msgs, on_error=on_error, stack=stack)

    def raises(_index):
        msg = msgs[_index.item()]
        E._jit.last_error_info = (msg, stack)
        raise E._EquinoxRuntimeError(msg)

    struct = jax.eval_shape(lambda: x)
    return jax.lax.cond(
        pred, lambda: jax.pure_callback(raises, struct, index), lambda: x)


def install(variant):
    EnumerationItem.error_if = _ORIGINAL_ERROR_IF
    E._error = _ORIGINAL_ERROR
    if variant == "v0_baseline":
        EnumerationItem.error_if = v0_baseline
    elif variant == "v3_one_cb":
        E._error = _error_single_callback
    elif variant == "v4_small":
        EnumerationItem.error_if = v4_small


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["v0_baseline", "v3_one_cb", "v4_small"])
    args = ap.parse_args()
    install(args.variant)

    a = dq.destroy(N)
    H = DELTA * dq.dag(a) @ a
    Ls = [np.sqrt(K2) * (a @ a - ALPHA**2 * dq.eye(N)), np.sqrt(K1) * a]
    rho0 = dq.fock_dm(N, 0).asdense()
    print(f"### {args.variant}   N={N}  R={R:,d} B", flush=True)

    def make(nt, max_steps=100_000):
        ts = jnp.linspace(0.0, T_FINAL, nt)
        return jax.jit(lambda r: dq.mesolve(
            H, Ls, r, ts,
            method=dq.method.Tsit5(rtol=1e-6, atol=1e-6, max_steps=max_steps),
            save_states=True, progress_meter=False))

    xs, outs, temps = [], [], []
    for nt in (10, 20, 40, 80, 160):
        m = make(nt).lower(rho0).compile().memory_analysis()
        xs.append(nt)
        outs.append(m.output_size_in_bytes)
        temps.append(m.temp_size_in_bytes)
        print(f"  ntsave={nt:>4}  output={m.output_size_in_bytes:>12,}  "
              f"temp={m.temp_size_in_bytes:>12,}  total/traj="
              f"{(m.output_size_in_bytes + m.temp_size_in_bytes) / (nt * R):.4f}",
              flush=True)
    x = np.array(xs, float)
    A = np.column_stack([np.ones_like(x), x])
    for name, y in (("output", outs), ("temp", temps),
                    ("total", np.array(outs) + np.array(temps))):
        b, *_ = np.linalg.lstsq(A, np.array(y, float), rcond=None)
        print(f"  {name:<7s} = {b[0]:>12,.0f} B + {b[1] / R:>7.4f} R per saved state",
              flush=True)

    print("  --- safety check: max_steps=1 must still raise ---", flush=True)
    try:
        out = jax.block_until_ready(make(11, max_steps=1)(rho0))
        st = np.asarray(out.states)
        print(f"  *** NO ERROR RAISED *** finite={np.isfinite(st).all()}", flush=True)
    except Exception as exc:  # noqa: BLE001
        first = str(exc).strip().split("\n")[0][:100]
        print(f"  OK, raised {type(exc).__name__}: {first}", flush=True)


if __name__ == "__main__":
    main()
