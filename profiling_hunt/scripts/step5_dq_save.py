"""Step 5 - does dynamiqs inherit the 3x trajectory peak of Finding 3.A(ii)?

Step 3a measured, in RAW diffrax with SaveAt(steps=True):

    peak = 2.000*S + 3.000*S*max_steps        (exact to +-16 B)

i.e. two EXTRA full copies of the trajectory buffer live at the high-water mark.
Step 4 switched trajectory storage off entirely, so dynamiqs was never tested on
this - yet save_states=True is its default. This closes that gap.

What reading dynamiqs' source (integrators/core/diffrax_integrator.py,
save_mixin.py) establishes before any measurement:

  * dynamiqs uses  dx.SubSaveAt(ts=self.ts, fn=fn)  - an EXPLICIT ts list, not
    SaveAt(steps=True). So trajectory storage should scale with len(tsave) and
    NOT with max_steps. Prediction: no max_steps sizing trap.
  * saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b]) where subsaveat_b is
    SubSaveAt(t1=True): one EXTRA state is always saved on top of the ts ones.
  * save(y) returns SolveSaved(ysave, extra, Esave) with ysave = y only if
    options.save_states. With save_states=False the per-ts payload has no array
    leaf at all, so resident trajectory should be exactly 1 state (the t1 one).
  * Esave = stack([expect(E, y) for E in Es]) -> (ntsave, nE) SCALARS, not N^2.
  * adjoint: gradient=None  -> dx.RecursiveCheckpointAdjoint()   <-- checkpointed
             Forward        -> dx.ForwardMode()
             Direct         -> dx.DirectAdjoint()
    A forward-only solve therefore runs inside a CHECKPOINTED while loop by
    default, which allocates binomial-checkpoint storage for a gradient that was
    never requested.
  * max_steps is HARD-CODED to 100_000 for fixed-step methods (there is a TODO),
    and is method.max_steps (default 100_000) for adaptive ones.

Hypotheses tested here:
  H-A  trajectory resident = ntsave * R  (+1 R for the t1 save)     [coefficient 1]
  H-B  peak = a + b * ntsave * R.  b = 3.000 would mean dynamiqs inherits the raw
       diffrax behaviour; b = 1.000 would mean its buffer handling avoids it.
  H-C  memory is independent of max_steps (explicit ts, not steps=True).
  H-D  gradient/adjoint choice changes forward-only memory.
"""

import argparse
import gc
import json
import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import dynamiqs as dq  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

GIB = 1024**3
MIB = 1024**2

ALPHA, K2, K1, DELTA = 2.0, 1.0, 0.05, 0.2
T_FINAL = 0.1

GRADIENTS = {
    "none": None,
    "forward": lambda: dq.gradient.Forward(),
    "direct": lambda: dq.gradient.Direct(),
    "checkpointed": lambda: dq.gradient.BackwardCheckpointed(),
}


def stats(dev):
    return dev.memory_stats()


def tree_bytes(x) -> int:
    return sum(int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree.leaves(x))


def build(N, layout):
    """Cat qubit operators; dense rho0 either way (the state is always dense)."""
    a = dq.destroy(N)
    H = DELTA * dq.dag(a) @ a
    L2 = np.sqrt(K2) * (a @ a - ALPHA**2 * dq.eye(N))
    L1 = np.sqrt(K1) * a
    if layout == "dense":
        H, L2, L1 = H.asdense(), L2.asdense(), L1.asdense()
    rho0 = dq.fock_dm(N, 0).asdense()
    return H, [L2, L1], rho0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--ntsave", type=int, default=11)
    ap.add_argument("--save-states", type=int, default=1)
    ap.add_argument("--n-exp-ops", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--gradient", type=str, default="none",
                    choices=list(GRADIENTS))
    ap.add_argument("--layout", type=str, default="dia", choices=["dia", "dense"])
    ap.add_argument("--json", type=str, required=True)
    args = ap.parse_args()

    dev = jax.local_devices()[0]
    N = args.N
    R = 16 * N * N
    save_states = bool(args.save_states)

    print("#" * 110, flush=True)
    print(f"STEP 5  dynamiqs {dq.__version__}  N={N}  R = 16*N^2 = {R:,d} B "
          f"({R / MIB:.2f} MiB)", flush=True)
    print(f"  ntsave={args.ntsave}  save_states={save_states}  "
          f"n_exp_ops={args.n_exp_ops}  max_steps={args.max_steps}  "
          f"gradient={args.gradient}  layout={args.layout}", flush=True)
    for v in ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE"):
        print(f"  {v} = {os.environ.get(v, '<unset>')}", flush=True)
    print(f"  bytes_limit = {stats(dev)['bytes_limit']:,d}", flush=True)

    pred_traj = (args.ntsave * R if save_states else R)
    print(f"  H-A PREDICTION trajectory resident = "
          f"{'ntsave' if save_states else '1 (t1 only)'} x R = {pred_traj:,d} B "
          f"= {pred_traj / GIB:.5f} GiB", flush=True)
    print("#" * 110, flush=True)

    rec = {"N": N, "R": R, "ntsave": args.ntsave, "save_states": save_states,
           "n_exp_ops": args.n_exp_ops, "max_steps": args.max_steps,
           "gradient": args.gradient, "layout": args.layout,
           "pred_traj_bytes": pred_traj, "status": "ok"}

    H = jump_ops = rho0 = res = Es = None
    try:
        H, jump_ops, rho0 = build(N, args.layout)
        Es = None
        if args.n_exp_ops > 0:
            a = dq.destroy(N)
            Es = [dq.dag(a) @ a, a + dq.dag(a)][: args.n_exp_ops]
        op_bytes = tree_bytes(H) + tree_bytes(jump_ops) + tree_bytes(Es)
        st0 = stats(dev)
        base, peak_setup = st0["bytes_in_use"], st0["peak_bytes_in_use"]
        print(f"  after setup: in_use={base:,d} ({base / R:.3f} R)  "
              f"peak={peak_setup:,d} ({peak_setup / R:.3f} R)  "
              f"op_bytes={op_bytes:,d}", flush=True)

        tsave = jnp.linspace(0.0, T_FINAL, args.ntsave)
        grad = GRADIENTS[args.gradient]
        grad = grad() if grad is not None else None

        t0 = time.perf_counter()
        res = dq.mesolve(
            H, jump_ops, rho0, tsave, exp_ops=Es,
            method=dq.method.Tsit5(rtol=1e-6, atol=1e-6, max_steps=args.max_steps),
            gradient=grad, save_states=save_states, progress_meter=False)
        jax.block_until_ready(res)
        el = time.perf_counter() - t0

        st = stats(dev)
        pk = st["peak_bytes_in_use"]
        states = getattr(res, "states", None)
        traj_bytes = tree_bytes(states)
        expects = getattr(res, "expects", None)
        exp_bytes = tree_bytes(expects)

        print(f"  after mesolve: in_use={st['bytes_in_use']:,d} "
              f"({st['bytes_in_use'] / R:.3f} R)  peak={pk:,d} ({pk / R:.3f} R)",
              flush=True)
        print(f"  res.states shape="
              f"{None if states is None else tuple(np.shape(states))}  "
              f"bytes={traj_bytes:,d} ({traj_bytes / R:.3f} R)", flush=True)
        print(f"  H-A measured/predicted trajectory = "
              f"{traj_bytes / pred_traj:.4f}", flush=True)
        print(f"  res.expects shape="
              f"{None if expects is None else tuple(np.shape(expects))}  "
              f"bytes={exp_bytes:,d}", flush=True)
        print(f"  peak during SETUP = {peak_setup:,d}; setup-dominated = "
              f"{pk == peak_setup}", flush=True)
        print(f"  elapsed {el:.1f}s", flush=True)

        rec.update({"op_bytes": op_bytes, "peak": pk, "peak_in_R": pk / R,
                    "in_use_after": st["bytes_in_use"], "traj_bytes": traj_bytes,
                    "traj_in_R": traj_bytes / R, "exp_bytes": exp_bytes,
                    "peak_setup": peak_setup,
                    "setup_dominated": pk == peak_setup,
                    "elapsed_s": round(el, 2), "num_allocs": st["num_allocs"]})
    except Exception as exc:  # noqa: BLE001 - OOM is an expected outcome
        st = stats(dev)
        print(f"  !! {type(exc).__name__}: {str(exc)[:400]}", flush=True)
        rec.update({"status": "OOM" if "RESOURCE_EXHAUSTED" in str(exc) else "ERR",
                    "exc": type(exc).__name__, "peak": st["peak_bytes_in_use"],
                    "peak_in_R": st["peak_bytes_in_use"] / R})
    finally:
        H = jump_ops = rho0 = res = Es = None
        gc.collect()
        jax.clear_caches()
        rec["clean"] = stats(dev)["bytes_in_use"] == 0

    with open(args.json, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"\n  appended record to {args.json}", flush=True)


if __name__ == "__main__":
    main()
