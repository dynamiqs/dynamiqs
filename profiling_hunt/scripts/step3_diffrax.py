"""Step 3 - Isolating diffrax: does solver memory scale the way the algorithm says?

Bare diffrax, no physics, no dynamiqs. The state is a MATRIX so that per-stage
copies are actually measurable:

    dY/dt = M @ Y ,   Y, M of shape (N, N), float64

    state_bytes = 8 * N^2                    [float64 = 8 bytes/element]

Two independent hypotheses, tested separately:

  3a  memory vs number of time steps   (Euler, fixed dt, SaveAt t1 vs steps)
  3b  memory vs number of solver stages (Ralston/Bosh3/Tsit5/Dopri8)

No jit is applied by this script. diffeqsolve internally traces its own loop body,
so 3b additionally drives solver.step() directly from a Python for-loop, which is
genuinely un-jitted at the call level and exposes per-step allocation.
"""

import argparse
import gc
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

GIB = 1024**3
MIB = 1024**2

SOLVERS = {
    "Euler": (diffrax.Euler, 1, 1),
    "Ralston": (diffrax.Ralston, 2, 2),
    "Bosh3": (diffrax.Bosh3, 3, 4),
    "Tsit5": (diffrax.Tsit5, 5, 7),
    "Dopri8": (diffrax.Dopri8, 8, 14),
}

METRIC_KEYS = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit", "pool_bytes",
               "peak_pool_bytes", "largest_alloc_size", "num_allocs")


def stats(dev) -> dict:
    return dev.memory_stats()


def show(dev, tag: str, base: int = 0) -> dict:
    st = stats(dev)
    print(f"    {tag:<38} in_use={st['bytes_in_use']:>15,d} "
          f"({st['bytes_in_use'] / GIB:8.5f} GiB)  "
          f"peak={st['peak_bytes_in_use']:>15,d} "
          f"({st['peak_bytes_in_use'] / GIB:8.5f} GiB)  "
          f"nalloc={st['num_allocs']:>7,d}"
          + (f"  delta_in_use={st['bytes_in_use'] - base:+,d} "
             f"({(st['bytes_in_use'] - base) / MIB:+.2f} MiB)" if base else ""),
          flush=True)
    return st


def make_problem(N: int, seed: int = 0):
    """dY/dt = M @ Y with a stable-ish M."""
    kM, kY = jax.random.split(jax.random.key(seed))
    M = jax.random.normal(kM, (N, N), dtype=jnp.float64) / (N**0.5)
    M = M - 0.5 * jnp.eye(N, dtype=jnp.float64)
    Y0 = jax.random.normal(kY, (N, N), dtype=jnp.float64)
    M.block_until_ready()
    Y0.block_until_ready()
    assert M.dtype == jnp.float64 and Y0.dtype == jnp.float64
    return M, Y0


# --------------------------------------------------------------------------- 3a
def run_3a(dev, N: int, n_steps: int, save_mode: str, max_steps_mult: float,
           out: list, vf_mode: str = "closure") -> None:
    """Euler, fixed dt, measure memory vs step count and SaveAt mode."""
    state_bytes = 8 * N * N
    max_steps = max(int(n_steps * max_steps_mult), n_steps + 1)

    if save_mode == "t1":
        saveat = diffrax.SaveAt(t1=True)
        pred_traj = state_bytes
        pred_desc = "1 x state (final only)"
    elif save_mode == "steps":
        saveat = diffrax.SaveAt(steps=True)
        pred_traj = max_steps * state_bytes
        pred_desc = f"max_steps({max_steps}) x state"
    else:
        raise ValueError(save_mode)

    print(f"\n  --- 3a  N={N} n_steps={n_steps} SaveAt={save_mode} "
          f"max_steps={max_steps} ---", flush=True)
    print(f"    state_bytes = 8*{N}^2 = {state_bytes:,d} B = "
          f"{state_bytes / MIB:.2f} MiB", flush=True)
    print(f"    PREDICTED trajectory storage: {pred_desc} = {pred_traj:,d} B = "
          f"{pred_traj / GIB:.5f} GiB", flush=True)

    M, Y0 = make_problem(N)
    base = show(dev, "after problem setup (M + Y0)")["bytes_in_use"]

    rec = {"part": "3a", "N": N, "n_steps": n_steps, "save": save_mode,
           "vf_mode": vf_mode,
           "max_steps": max_steps, "state_bytes": state_bytes,
           "pred_traj_bytes": pred_traj, "status": "ok", "exc": ""}
    sol = None
    try:
        if vf_mode == "closure":
            def vf(t, y, args):
                return M @ y          # M captured by the closure
            solve_args = None
        else:
            def vf(t, y, args):
                return args @ y       # M passed in as an argument
            solve_args = M

        term = diffrax.ODETerm(vf)
        t0, t1 = 0.0, 1.0
        dt0 = (t1 - t0) / n_steps
        t_start = time.perf_counter()
        sol = diffrax.diffeqsolve(
            term, diffrax.Euler(), t0=t0, t1=t1, dt0=dt0, y0=Y0, args=solve_args,
            saveat=saveat, max_steps=max_steps,
            stepsize_controller=diffrax.ConstantStepSize(),
        )
        jax.block_until_ready(sol)
        dt = time.perf_counter() - t_start
        st = show(dev, "after diffeqsolve", base)
        ys_bytes = int(sol.ys.nbytes) if sol.ys is not None else 0
        print(f"    sol.ys shape={None if sol.ys is None else sol.ys.shape} "
              f"nbytes={ys_bytes:,d} ({ys_bytes / GIB:.5f} GiB)", flush=True)
        print(f"    measured trajectory / predicted = "
              f"{ys_bytes / pred_traj:.4f}", flush=True)
        print(f"    elapsed {dt:.2f}s", flush=True)
        rec.update({m: st[m] for m in METRIC_KEYS})
        rec.update({"ys_bytes": ys_bytes, "delta_in_use": st["bytes_in_use"] - base,
                    "elapsed_s": round(dt, 3)})
    except Exception as exc:  # noqa: BLE001 - OOM expected, keep going
        st = stats(dev)
        print(f"    !! {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        rec.update({"status": "OOM" if "RESOURCE_EXHAUSTED" in str(exc) else "ERR",
                    "exc": type(exc).__name__})
        rec.update({m: st[m] for m in METRIC_KEYS})
    finally:
        # Every local holding a device buffer must go, including the closure that
        # captured M and the ODETerm that holds the closure.
        sol = M = Y0 = solve_args = None
        term = vf = saveat = None
        gc.collect()
        jax.clear_caches()
        after = stats(dev)
        print(f"    after cleanup: in_use={after['bytes_in_use']:,d} "
              f"(baseline return: "
              f"{'YES' if after['bytes_in_use'] == 0 else 'NO <-- DIRTY'})", flush=True)
        rec["clean"] = after["bytes_in_use"] == 0
    out.append(rec)


# --------------------------------------------------------------------------- 3b
def run_3b_manual(dev, N: int, solver_name: str, n_steps: int, out: list) -> None:
    """Drive solver.step() from a Python loop - no jit at the call level."""
    cls, order, stages = SOLVERS[solver_name]
    state_bytes = 8 * N * N
    print(f"\n  --- 3b MANUAL  N={N} solver={solver_name} order={order} "
          f"stages={stages} n_steps={n_steps} ---", flush=True)
    print(f"    state_bytes = {state_bytes:,d} B = {state_bytes / MIB:.2f} MiB",
          flush=True)
    print(f"    PREDICTION: peak grows as a + b*n_stages with b ~ state_bytes "
          f"=> {stages} x {state_bytes / MIB:.2f} MiB = "
          f"{stages * state_bytes / MIB:.2f} MiB of stage storage", flush=True)

    M, Y0 = make_problem(N)
    base = show(dev, "after problem setup (M + Y0)")["bytes_in_use"]

    rec = {"part": "3b_manual", "N": N, "solver": solver_name, "order": order,
           "stages": stages, "n_steps": n_steps, "state_bytes": state_bytes,
           "status": "ok", "exc": ""}
    y = None
    try:
        def vf(t, y_, args):
            return M @ y_

        term = diffrax.ODETerm(vf)
        solver = cls()
        t0, t1 = 0.0, 1.0
        dt = (t1 - t0) / n_steps
        solver_state = solver.init(term, t0, t0 + dt, Y0, None)
        jax.block_until_ready(solver_state)
        after_init = show(dev, "after solver.init", base)

        y = Y0
        peaks, inuses, nallocs = [], [], []
        for i in range(n_steps):
            ta = t0 + i * dt
            tb = ta + dt
            y, y_err, dense_info, solver_state, result = solver.step(
                term, ta, tb, y, None, solver_state, made_jump=False)
            jax.block_until_ready((y, y_err, solver_state))
            st = stats(dev)
            peaks.append(st["peak_bytes_in_use"])
            inuses.append(st["bytes_in_use"])
            nallocs.append(st["num_allocs"])
            print(f"    step {i:>3}: in_use={st['bytes_in_use']:>14,d} "
                  f"({st['bytes_in_use'] / MIB:9.2f} MiB = "
                  f"{st['bytes_in_use'] / state_bytes:6.2f} states)  "
                  f"peak={st['peak_bytes_in_use']:>14,d} "
                  f"({st['peak_bytes_in_use'] / state_bytes:6.2f} states)  "
                  f"nalloc={st['num_allocs']:>7,d}", flush=True)
        st = show(dev, "after all steps", base)
        peak = max(peaks)
        stage_budget = peak - after_init["bytes_in_use"]
        print(f"    peak_bytes_in_use over the loop = {peak:,d} "
              f"({peak / MIB:.2f} MiB)", flush=True)
        print(f"    peak - post_init_resident      = {stage_budget:,d} "
              f"({stage_budget / MIB:.2f} MiB) = "
              f"{stage_budget / state_bytes:.3f} x state", flush=True)
        # Leak test: does resident memory grow with the step index? Fit a line to
        # bytes_in_use vs step number over the second half (the first step or two
        # include one-off warm-up allocations).
        half = inuses[len(inuses) // 2:]
        if len(half) >= 2:
            xs = list(range(len(half)))
            mx = sum(xs) / len(xs)
            my = sum(half) / len(half)
            den = sum((x - mx) ** 2 for x in xs)
            slope = (sum((x - mx) * (y_ - my) for x, y_ in zip(xs, half)) / den
                     if den else 0.0)
        else:
            slope = 0.0
        print(f"    LEAK TEST: bytes_in_use slope over the last "
              f"{len(half)} steps = {slope:+,.1f} B/step = "
              f"{slope / state_bytes:+.4f} states/step", flush=True)
        print(f"    num_allocs per step = "
              f"{(nallocs[-1] - nallocs[0]) / max(len(nallocs) - 1, 1):.1f}",
              flush=True)
        rec.update({m: st[m] for m in METRIC_KEYS})
        rec.update({"peak_over_loop": peak,
                    "post_init_in_use": after_init["bytes_in_use"],
                    "stage_budget": stage_budget,
                    "stage_budget_in_states": stage_budget / state_bytes,
                    "in_use_end": st["bytes_in_use"],
                    "in_use_end_in_states": (st["bytes_in_use"]) / state_bytes,
                    "inuse_series": inuses, "peak_series": peaks,
                    "nalloc_series": nallocs,
                    "leak_slope_bytes_per_step": slope,
                    "leak_slope_states_per_step": slope / state_bytes,
                    "allocs_per_step": (nallocs[-1] - nallocs[0])
                    / max(len(nallocs) - 1, 1)})
    except Exception as exc:  # noqa: BLE001
        st = stats(dev)
        print(f"    !! {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        rec.update({"status": "OOM" if "RESOURCE_EXHAUSTED" in str(exc) else "ERR",
                    "exc": type(exc).__name__})
        rec.update({m: st[m] for m in METRIC_KEYS})
    finally:
        y = M = Y0 = None
        term = vf = solver = solver_state = None
        y_err = dense_info = result = None
        gc.collect()
        jax.clear_caches()
        after = stats(dev)
        print(f"    after cleanup: in_use={after['bytes_in_use']:,d} "
              f"(baseline return: "
              f"{'YES' if after['bytes_in_use'] == 0 else 'NO <-- DIRTY'})", flush=True)
        rec["clean"] = after["bytes_in_use"] == 0
    out.append(rec)


def run_3b_solve(dev, N: int, solver_name: str, n_steps: int, out: list,
                 vf_mode: str = "closure") -> None:
    """Same sweep through the real diffeqsolve API, SaveAt(t1=True)."""
    cls, order, stages = SOLVERS[solver_name]
    state_bytes = 8 * N * N
    print(f"\n  --- 3b DIFFEQSOLVE  N={N} solver={solver_name} stages={stages} "
          f"n_steps={n_steps} ---", flush=True)

    M, Y0 = make_problem(N)
    base = show(dev, "after problem setup (M + Y0)")["bytes_in_use"]
    rec = {"part": "3b_solve", "N": N, "solver": solver_name, "order": order,
           "vf_mode": vf_mode,
           "stages": stages, "n_steps": n_steps, "state_bytes": state_bytes,
           "status": "ok", "exc": ""}
    sol = None
    try:
        if vf_mode == "closure":
            def vf(t, y_, args):
                return M @ y_
            solve_args = None
        else:
            def vf(t, y_, args):
                return args @ y_
            solve_args = M

        term = diffrax.ODETerm(vf)
        t0, t1 = 0.0, 1.0
        t_start = time.perf_counter()
        sol = diffrax.diffeqsolve(
            term, cls(), t0=t0, t1=t1, dt0=(t1 - t0) / n_steps, y0=Y0,
            args=solve_args,
            saveat=diffrax.SaveAt(t1=True), max_steps=n_steps + 1,
            stepsize_controller=diffrax.ConstantStepSize(),
        )
        jax.block_until_ready(sol)
        el = time.perf_counter() - t_start
        st = show(dev, "after diffeqsolve", base)
        print(f"    peak/state = {st['peak_bytes_in_use'] / state_bytes:.3f}   "
              f"elapsed {el:.2f}s", flush=True)
        rec.update({m: st[m] for m in METRIC_KEYS})
        rec.update({"peak_in_states": st["peak_bytes_in_use"] / state_bytes,
                    "elapsed_s": round(el, 3)})
    except Exception as exc:  # noqa: BLE001
        st = stats(dev)
        print(f"    !! {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        rec.update({"status": "OOM" if "RESOURCE_EXHAUSTED" in str(exc) else "ERR",
                    "exc": type(exc).__name__})
        rec.update({m: st[m] for m in METRIC_KEYS})
    finally:
        sol = M = Y0 = solve_args = None
        term = vf = None
        gc.collect()
        jax.clear_caches()
        after = stats(dev)
        print(f"    after cleanup: in_use={after['bytes_in_use']:,d} "
              f"(baseline return: "
              f"{'YES' if after['bytes_in_use'] == 0 else 'NO <-- DIRTY'})", flush=True)
        rec["clean"] = after["bytes_in_use"] == 0
    out.append(rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["3a", "3b_manual", "3b_solve"])
    ap.add_argument("--N", type=int, default=4000)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--solver", type=str, default="Tsit5")
    ap.add_argument("--save", type=str, default="t1", choices=["t1", "steps"])
    ap.add_argument("--max-steps-mult", type=float, default=1.0)
    ap.add_argument("--vf-mode", type=str, default="closure",
                    choices=["closure", "args"],
                    help="does the vector field capture M, or receive it via args?")
    ap.add_argument("--json", type=str, required=True)
    args = ap.parse_args()

    dev = jax.local_devices()[0]
    st = stats(dev)
    print("#" * 110, flush=True)
    print(f"STEP 3 [{args.part}]  jax {jax.__version__} diffrax {diffrax.__version__} "
          f"x64={jax.config.jax_enable_x64}", flush=True)
    for v in ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE"):
        print(f"  {v} = {os.environ.get(v, '<unset>')}", flush=True)
    print(f"  bytes_limit = {st['bytes_limit']:,d} B = {st['bytes_limit'] / GIB:.4f} GiB",
          flush=True)
    print(f"  float64 = 8 bytes/element; state_bytes = 8*N^2", flush=True)
    print("#" * 110, flush=True)

    out: list = []
    if args.part == "3a":
        run_3a(dev, args.N, args.n_steps, args.save, args.max_steps_mult, out,
               args.vf_mode)
    elif args.part == "3b_manual":
        run_3b_manual(dev, args.N, args.solver, args.n_steps, out)
    else:
        run_3b_solve(dev, args.N, args.solver, args.n_steps, out, args.vf_mode)

    with open(args.json, "a") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"\n  appended {len(out)} record(s) to {args.json}", flush=True)


if __name__ == "__main__":
    main()
