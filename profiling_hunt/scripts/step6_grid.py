"""Step 6 - grid-search validation of the two Step-5 memory formulas, with R^2.

The two formulas under test (from profiling_hunt.md, Findings 5.A / 5.C / 4.A / 3.B),
with R = 16 N^2 bytes (one complex128 density matrix) and S = number of RK stages:

    (1) save_states=False :  peak = (C0 + S) * R                          [flat in ntsave]
    (2) save_states=True  :  peak = max( (C0 + S) * R , 3 * R * (ntsave + 1) )
                             + 1 * R * ntsave   if gradient=Direct

Every point in the grid is measured with `peak_bytes_in_use`, the metric Step 2
selected as the OOM predictor. One CONFIG PER PROCESS, because peak_bytes_in_use is
a high-water mark that never resets.

Everything runs through the FULL dynamiqs library (dq.mesolve): its Lindblad vector
field, its DIA qarrays, its SubSaveAt(ts=...) save path, its adjoint selection. The
only thing monkey-patched is the diffrax RK *tableau object* inside dynamiqs' own
integrator constructor, because dynamiqs 0.3.6 exposes no adaptive RK2/RK3 method
(its adaptive set is Dopri5/Dopri8/Tsit5/Kvaerno3/Kvaerno5, and the Kvaerno methods
are implicit -- they would materialise an N^2 x N^2 Jacobian and OOM for reasons
having nothing to do with these formulas). fixed_step stays False, so the
PIDController adaptive stepsize controller is used for every method: adaptive only,
as required.

    rk2 -> dx.Heun()    order 2,  2 stages   (tableau swapped into the Dopri5 slot)
    rk3 -> dx.Bosh3()   order 3,  4 stages   (tableau swapped into the Dopri5 slot)
    rk5 -> dx.Tsit5()   order 5,  7 stages   (native dq.method.Tsit5, no patch)
    rk8 -> dx.Dopri8()  order 8, 14 stages   (native dq.method.Dopri8, no patch)
"""

import argparse
import gc
import json
import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import diffrax as dx  # noqa: E402
import dynamiqs as dq  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from dynamiqs.integrators.apis import mesolve as _mesolve_api  # noqa: E402
from dynamiqs.integrators.core.diffrax_integrator import (  # noqa: E402
    MESolveDiffraxIntegrator,
)
from dynamiqs.result import MESolveResult  # noqa: E402
from functools import partial  # noqa: E402

GIB = 1024**3
MIB = 1024**2

# cat qubit, identical physics to Step 5 so the two studies are comparable
ALPHA, K2, K1, DELTA = 2.0, 1.0, 0.05, 0.2
T_FINAL = 0.1
RTOL = ATOL = 1e-6

# label -> (diffrax tableau, n_stages, order, dynamiqs method class used to reach it)
METHODS = {
    "rk2": (dx.Heun(), 2, 2, "Dopri5"),
    "rk3": (dx.Bosh3(), 4, 3, "Dopri5"),
    "rk5": (dx.Tsit5(), 7, 5, "Tsit5"),
    "rk8": (dx.Dopri8(), 14, 8, "Dopri8"),
}

GRADIENTS = {"none": None, "direct": lambda: dq.gradient.Direct()}


def install_method(label, max_steps):
    """Return a dq.method instance, patching the RK tableau only when needed."""
    tableau, n_stages, order, slot = METHODS[label]
    cls = getattr(dq.method, slot)
    if label in ("rk2", "rk3"):
        # swap ONLY the tableau object in dynamiqs' own constructor; fixed_step stays
        # False so dynamiqs still builds a dx.PIDController (adaptive).
        _mesolve_api.mesolve_dopri5_integrator_constructor = partial(
            MESolveDiffraxIntegrator,
            diffrax_solver=tableau,
            fixed_step=False,
            result_class=MESolveResult,
        )
    return cls(rtol=RTOL, atol=ATOL, max_steps=max_steps), n_stages, order


def tree_bytes(x) -> int:
    return sum(int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree.leaves(x))


def build(N):
    """Cat qubit with native DIA operators (the formula assumes DIA throughout)."""
    a = dq.destroy(N)
    H = DELTA * dq.dag(a) @ a
    L2 = np.sqrt(K2) * (a @ a - ALPHA**2 * dq.eye(N))
    L1 = np.sqrt(K1) * a
    rho0 = dq.fock_dm(N, 0).asdense()
    return H, [L2, L1], rho0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--ntsave", type=int, required=True)
    ap.add_argument("--save-states", type=int, required=True)
    ap.add_argument("--method", type=str, required=True, choices=list(METHODS))
    ap.add_argument("--gradient", type=str, required=True, choices=list(GRADIENTS))
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--json", type=str, required=True)
    args = ap.parse_args()

    dev = jax.local_devices()[0]
    N, nt = args.N, args.ntsave
    R = 16 * N * N
    save_states = bool(args.save_states)
    is_direct = args.gradient == "direct"

    method, n_stages, order = install_method(args.method, args.max_steps)
    tableau = METHODS[args.method][0]
    # the dense-interpolation class is recorded because SubSaveAt(ts=...) interpolates
    # rather than stepping to each save point, so it is a second, method-dependent
    # consumer of state-sized buffers alongside the RK stages themselves
    interp = getattr(type(tableau), "interpolation_cls", None)
    interp = getattr(interp, "__name__", str(interp))

    # the formula's own prediction, computed BEFORE the run
    C0 = 7.0                      # calibrated at nL=2 in profiling_hunt.md 5.2
    floor = (C0 + n_stages) * R
    if save_states:
        pred = max(floor, 3.0 * R * (nt + 1) + (1.0 * R * nt if is_direct else 0.0))
    else:
        pred = floor

    print("#" * 112, flush=True)
    print(f"STEP 6 GRID  dynamiqs {dq.__version__}  N={N}  R=16N^2={R:,d} B "
          f"({R / MIB:.3f} MiB)", flush=True)
    print(f"  method={args.method} (order {order}, {n_stages} stages, adaptive PID)  "
          f"ntsave={nt}  save_states={save_states}  gradient={args.gradient}",
          flush=True)
    for v in ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE"):
        print(f"  {v} = {os.environ.get(v, '<unset>')}", flush=True)
    st0 = dev.memory_stats()
    print(f"  bytes_limit = {st0['bytes_limit']:,d}", flush=True)
    print(f"  FORMULA PREDICTION peak = {pred:,.0f} B = {pred / GIB:.5f} GiB "
          f"= {pred / R:.3f} R", flush=True)
    print("#" * 112, flush=True)

    rec = {"N": N, "R": R, "ntsave": nt, "save_states": save_states,
           "method": args.method, "n_stages": n_stages, "order": order,
           "gradient": args.gradient, "is_direct": is_direct,
           "interp": interp, "n_jump": 2, "solver": type(tableau).__name__,
           "max_steps": args.max_steps, "bytes_limit": st0["bytes_limit"],
           "pred_peak": pred, "status": "ok"}

    H = jump_ops = rho0 = res = None
    try:
        H, jump_ops, rho0 = build(N)
        op_bytes = tree_bytes(H) + tree_bytes(jump_ops)
        stx = dev.memory_stats()
        print(f"  after setup: in_use={stx['bytes_in_use']:,d} "
              f"peak={stx['peak_bytes_in_use']:,d}  op_bytes={op_bytes:,d} "
              f"({op_bytes / R:.5f} R)", flush=True)

        tsave = jnp.linspace(0.0, T_FINAL, nt)
        g = GRADIENTS[args.gradient]
        g = g() if g is not None else None

        t0 = time.perf_counter()
        res = dq.mesolve(H, jump_ops, rho0, tsave, method=method, gradient=g,
                         save_states=save_states, progress_meter=False)
        jax.block_until_ready(res)
        el = time.perf_counter() - t0

        st = dev.memory_stats()
        pk = st["peak_bytes_in_use"]
        states = getattr(res, "states", None)
        traj = tree_bytes(states)
        nsteps = None
        try:
            nsteps = int(np.asarray(res.infos.nsteps).reshape(-1)[0])
        except Exception:  # noqa: BLE001
            pass

        print(f"  after mesolve: in_use={st['bytes_in_use']:,d}  "
              f"peak={pk:,d} ({pk / R:.3f} R, {pk / GIB:.5f} GiB)", flush=True)
        print(f"  traj bytes={traj:,d} ({traj / R:.3f} R)  nsteps={nsteps}", flush=True)
        print(f"  MEASURED / PREDICTED = {pk / pred:.4f}   "
              f"rel.err = {pk / pred - 1:+.2%}", flush=True)
        print(f"  elapsed {el:.1f}s", flush=True)

        rec.update({"op_bytes": op_bytes, "peak": pk, "peak_in_R": pk / R,
                    "in_use_after": st["bytes_in_use"], "traj_bytes": traj,
                    "traj_in_R": traj / R, "peak_setup": stx["peak_bytes_in_use"],
                    "nsteps": nsteps, "num_allocs": st["num_allocs"],
                    "ratio_meas_pred": pk / pred, "elapsed_s": round(el, 2)})
    except Exception as exc:  # noqa: BLE001 - OOM is an expected outcome
        st = dev.memory_stats()
        msg = str(exc)
        oom = "RESOURCE_EXHAUSTED" in msg or "Out of memory" in msg
        print(f"  !! {type(exc).__name__}: {msg[:300]}", flush=True)
        rec.update({"status": "OOM" if oom else "ERR", "exc": type(exc).__name__,
                    "err": msg[:300], "peak": st["peak_bytes_in_use"],
                    "peak_in_R": st["peak_bytes_in_use"] / R})
    finally:
        H = jump_ops = rho0 = res = None
        states = traj = tsave = g = method = None
        gc.collect()
        jax.clear_caches()
        gc.collect()
        rec["in_use_final"] = dev.memory_stats()["bytes_in_use"]
        rec["clean"] = rec["in_use_final"] == 0

    with open(args.json, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"  -> appended to {args.json}", flush=True)


if __name__ == "__main__":
    main()
