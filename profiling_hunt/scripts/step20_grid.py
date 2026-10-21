"""Step 20 - dynamiqs-only memory laws for the save path, one config per process.

Three studies share this worker; everything else (physics, solver, tolerances) is
identical to Step 6 so the numbers are comparable.

  Study A (save_states=True, no exp_ops)
      Recover the trajectory law from a clean 5 x 5 grid in (N, ntsave).
      Step 6 claim under test:  peak = max(K, 3 R (ntsave + 1)).

  Study B (save_states=False, with exp_ops)
      What do observables cost once the trajectory is not stored? Axes:
      N, ntsave, n_obs, and the observable layout (dense vs DIA). The operators
      are the SAME quadratures x_phi in both layouts (.asdense() of the DIA ones),
      so the layout axis changes storage and code path and nothing else.
      Structural reason to expect a cost: with save_states=False and no exp_ops the
      SubSaveAt(ts=..., fn=save) payload has no array leaf, so the interpolated
      state y(ts) is dead code. Any exp_op revives it (expect = tracemm(E, y(ts)) =
      sum(E * y.T)), so the save callback now runs on a live state-sized value.

  Study C (save_states=False, boundary scan)
      Locates the edges of the window of N in which the save_states=False working
      set is ~16 R instead of ~14 R (seen in Study B at N=2560 and 3072), with and
      without one observable.

R = 16 N^2 bytes (one complex128 density matrix). Method: Tsit5 (S = 7), adaptive
PID, rtol = atol = 1e-6. No gradient. peak_bytes_in_use is a high-water mark that
never resets, hence ONE CONFIG PER PROCESS.
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

# cat qubit, identical physics to Steps 5 and 6
ALPHA, K2, K1, DELTA = 2.0, 1.0, 0.05, 0.2
T_FINAL = 0.1
RTOL = ATOL = 1e-6
N_STAGES = 7  # Tsit5


def tree_bytes(x) -> int:
    return sum(int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree.leaves(x))


def build(N):
    a = dq.destroy(N, layout=dq.dia)
    H = DELTA * a.dag() @ a
    L2 = np.sqrt(K2) * (a @ a - ALPHA**2 * dq.eye(N, layout=dq.dia))
    L1 = np.sqrt(K1) * a
    rho0 = dq.fock_dm(N, 0).asdense()
    return H, [L2, L1], rho0


def build_observables(N, n_obs, layout):
    """n_obs quadratures x_phi at evenly spaced angles, DIA or densified."""
    if n_obs == 0:
        return None
    ops = [dq.quadrature(N, np.pi * k / n_obs, layout=dq.dia) for k in range(n_obs)]
    if layout == "dense":
        ops = [op.asdense() for op in ops]
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=str, required=True, choices=["A", "B", "C"])
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--ntsave", type=int, required=True)
    ap.add_argument("--save-states", type=int, required=True)
    ap.add_argument("--n-obs", type=int, required=True)
    ap.add_argument("--obs-layout", type=str, required=True,
                    choices=["none", "dense", "dia"])
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument("--json", type=str, required=True)
    args = ap.parse_args()

    N, nt, n_obs = args.N, args.ntsave, args.n_obs
    R = 16 * N * N
    save_states = bool(args.save_states)
    rec = {"study": args.study, "N": N, "R": R, "ntsave": nt,
           "save_states": save_states, "n_obs": n_obs, "obs_layout": args.obs_layout,
           "method": "Tsit5", "n_stages": N_STAGES, "max_steps": args.max_steps,
           "status": "ok"}

    dev = jax.local_devices()[0]
    # GPU4 on higgs reports "Unknown Error"; JAX would silently fall back to CPU,
    # where memory_stats() is None and every number below would be meaningless
    if dev.platform != "gpu" or dev.memory_stats() is None:
        rec.update({"status": "ERR", "err": f"no usable GPU (platform={dev.platform})"})
        with open(args.json, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        raise SystemExit(f"no usable GPU: {dev}")

    st0 = dev.memory_stats()
    rec["bytes_limit"] = st0["bytes_limit"]
    print("#" * 100, flush=True)
    print(f"STEP 20 study {args.study}  dynamiqs {dq.__version__}  N={N}  "
          f"R={R:,d} B ({R / MIB:.3f} MiB)  ntsave={nt}  save_states={save_states}  "
          f"n_obs={n_obs} ({args.obs_layout})", flush=True)
    for v in ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE"):
        print(f"  {v} = {os.environ.get(v, '<unset>')}", flush=True)
    print(f"  bytes_limit = {st0['bytes_limit']:,d}", flush=True)

    H = jump_ops = rho0 = exp_ops = res = tsave = None
    try:
        H, jump_ops, rho0 = build(N)
        exp_ops = build_observables(N, n_obs, args.obs_layout)
        tsave = jnp.linspace(0.0, T_FINAL, nt)
        jax.block_until_ready((H, jump_ops, rho0, exp_ops, tsave))
        stx = dev.memory_stats()
        op_bytes = tree_bytes(H) + tree_bytes(jump_ops)
        obs_bytes = tree_bytes(exp_ops)
        print(f"  before mesolve: in_use={stx['bytes_in_use']:,d} "
              f"peak={stx['peak_bytes_in_use']:,d}  op_bytes={op_bytes:,d}  "
              f"obs_bytes={obs_bytes:,d} ({obs_bytes / R:.4f} R)", flush=True)

        t0 = time.perf_counter()
        res = dq.mesolve(H, jump_ops, rho0, tsave, exp_ops=exp_ops,
                         method=dq.method.Tsit5(rtol=RTOL, atol=ATOL,
                                                max_steps=args.max_steps),
                         save_states=save_states, progress_meter=False)
        jax.block_until_ready(res)
        elapsed = time.perf_counter() - t0

        st = dev.memory_stats()
        pk = st["peak_bytes_in_use"]
        traj = tree_bytes(res.states)
        expects = tree_bytes(res.expects)
        nsteps = None
        try:
            nsteps = int(np.asarray(res.infos.nsteps).reshape(-1)[0])
        except Exception:  # noqa: BLE001
            pass
        print(f"  after mesolve: peak={pk:,d} ({pk / R:.3f} R, {pk / GIB:.4f} GiB)  "
              f"traj={traj / R:.3f} R  expects={expects:,d} B  nsteps={nsteps}  "
              f"{elapsed:.1f}s", flush=True)

        rec.update({"peak": pk, "peak_in_R": pk / R,
                    "in_use_before": stx["bytes_in_use"],
                    "peak_setup": stx["peak_bytes_in_use"],
                    "op_bytes": op_bytes, "obs_bytes": obs_bytes,
                    "in_use_after": st["bytes_in_use"], "traj_bytes": traj,
                    "expects_bytes": expects, "nsteps": nsteps,
                    "elapsed_s": round(elapsed, 2)})
    except Exception as exc:  # noqa: BLE001 - OOM is an expected outcome
        st = dev.memory_stats()
        msg = str(exc)
        oom = "RESOURCE_EXHAUSTED" in msg or "Out of memory" in msg
        print(f"  !! {type(exc).__name__}: {msg[:300]}", flush=True)
        rec.update({"status": "OOM" if oom else "ERR", "exc": type(exc).__name__,
                    "err": msg[:300], "peak": st["peak_bytes_in_use"],
                    "peak_in_R": st["peak_bytes_in_use"] / R})
    finally:
        H = jump_ops = rho0 = exp_ops = res = tsave = None
        gc.collect()
        jax.clear_caches()
        gc.collect()
        rec["in_use_final"] = dev.memory_stats()["bytes_in_use"]

    with open(args.json, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"  -> appended to {args.json}", flush=True)


if __name__ == "__main__":
    main()
