"""Step 8 - can the error check thread on a small marker instead of the trajectory?

equinox's own API already calls the payload a `token`: "the token to thread the error
on to. This should be a PyTree containing at least one array." Diffrax hands it the
entire Solution. That is the whole cost: the check needs one array to hang a data
dependency on, and it is given the largest array in the program.

This tests four ways of writing `sol = result.error_if(sol, ...)`, by patching
EnumerationItem.error_if (the exact method diffrax calls) and measuring what XLA then
says it will allocate. Nothing else in diffrax or dynamiqs is touched.

  v0 baseline   thread on the whole Solution                      (what ships today)
  v1 marker     thread on the scalar result code, return sol as-is
                -> cheap, but the check is now dead code if the caller drops the code
  v2 barrier    thread on the scalar, then lax.optimization_barrier to tie the
                trajectory to the marker without passing it through the callback
  v3 one-cb     baseline payload, but drop the second (TPU only) callback

A variant is only useful if it is BOTH cheap AND still raises, so each one is also run
against a solve forced to fail with max_steps=1.
"""

import argparse
import sys

import jax

jax.config.update("jax_enable_x64", True)

import dynamiqs as dq  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.tree_util as jtu  # noqa: E402
import numpy as np  # noqa: E402
from equinox._enum import EnumerationItem  # noqa: E402
from equinox._errors import branched_error_if  # noqa: E402

MIB = 1024**2
N = 64
R = 16 * N * N
ALPHA, K2, K1, DELTA, T_FINAL = 2.0, 1.0, 0.05, 0.2, 0.1

_ORIGINAL = EnumerationItem.error_if


def v0_baseline(self, token, pred):
    return _ORIGINAL(self, token, pred)


def v1_marker(self, token, pred):
    # thread the error on the scalar result code; the payload never enters the callback
    branched_error_if(
        self._value, pred, self._value, self._enumeration._index_to_message
    )
    return token


def v2_barrier(self, token, pred):
    # thread on the scalar, then tie the payload to the marker with a pass-through
    # barrier, so the callback cannot be eliminated even though nothing large flows
    # through it
    marker = branched_error_if(
        self._value, pred, self._value, self._enumeration._index_to_message
    )
    leaves, treedef = jtu.tree_flatten(token)
    arrays = [x for x in leaves if eqx.is_array(x)]
    rest = [None if eqx.is_array(x) else x for x in leaves]
    tied, _ = jax.lax.optimization_barrier((arrays, marker))
    it = iter(tied)
    merged = [next(it) if r is None else r for r in rest]
    return jtu.tree_unflatten(treedef, merged)


def v3_one_callback(self, token, pred):
    # baseline, but with the second TPU-only callback removed
    import equinox._errors as E

    original_error = E._error

    def patched(x, pred_, index, *, msgs, on_error, stack):
        if on_error == "raise":
            def raises(_index):
                msg = msgs[_index.item()]
                E._jit.last_error_info = (msg, stack)
                raise E._EquinoxRuntimeError(msg)

            struct = jax.eval_shape(lambda: x)
            return jax.lax.cond(
                pred_, lambda: jax.pure_callback(raises, struct, index), lambda: x
            )
        return original_error(x, pred_, index, msgs=msgs, on_error=on_error, stack=stack)

    E._error.__wrapped__ if hasattr(E._error, "__wrapped__") else None
    saved = E._error
    E._error = patched
    try:
        return _ORIGINAL(self, token, pred)
    finally:
        E._error = saved


VARIANTS = {"v0_baseline": v0_baseline, "v1_marker": v1_marker,
            "v2_barrier": v2_barrier, "v3_one_callback": v3_one_callback}


def build():
    a = dq.destroy(N)
    H = DELTA * dq.dag(a) @ a
    Ls = [np.sqrt(K2) * (a @ a - ALPHA**2 * dq.eye(N)), np.sqrt(K1) * a]
    return H, Ls, dq.fock_dm(N, 0).asdense()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    args = ap.parse_args()
    EnumerationItem.error_if = VARIANTS[args.variant]

    H, Ls, rho0 = build()
    print(f"### {args.variant}   N={N}  R={R:,d} B", flush=True)

    def compile_for(nt, max_steps=100_000):
        ts = jnp.linspace(0.0, T_FINAL, nt)

        def f(rho0):
            return dq.mesolve(
                H, Ls, rho0, ts,
                method=dq.method.Tsit5(rtol=1e-6, atol=1e-6, max_steps=max_steps),
                save_states=True, progress_meter=False)

        return jax.jit(f), ts

    xs, outs, temps = [], [], []
    for nt in (10, 20, 40, 80, 160):
        fn, _ = compile_for(nt)
        m = fn.lower(rho0).compile().memory_analysis()
        xs.append(nt)
        outs.append(m.output_size_in_bytes)
        temps.append(m.temp_size_in_bytes)
        print(f"  ntsave={nt:>4}  output={m.output_size_in_bytes:>12,}  "
              f"temp={m.temp_size_in_bytes:>12,}  "
              f"total/traj={(m.output_size_in_bytes + m.temp_size_in_bytes) / (nt * R):.4f}",
              flush=True)
    x = np.array(xs, float)
    A = np.column_stack([np.ones_like(x), x])
    for name, y in (("output", outs), ("temp", temps),
                    ("total", np.array(outs) + np.array(temps))):
        b, *_ = np.linalg.lstsq(A, np.array(y, float), rcond=None)
        print(f"  {name:<7s} = {b[0]:>12,.0f} B + {b[1] / R:>7.4f} R per saved state",
              flush=True)

    # does it still raise when the solve genuinely fails?
    print("  --- safety: max_steps=1, must still raise ---", flush=True)
    fn, _ = compile_for(11, max_steps=1)
    try:
        out = jax.block_until_ready(fn(rho0))
        states = np.asarray(out.states)
        print(f"  NO ERROR RAISED. finite={np.isfinite(states).all()}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  raised {type(exc).__name__}: {str(exc)[:110]}", flush=True)


if __name__ == "__main__":
    main()
