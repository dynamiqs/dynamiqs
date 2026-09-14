"""Step 4 - Attributing the overhead: what do Lindblad and dynamiqs each add?

Four rows, all at the SAME state dimension and the SAME solver (Tsit5), so only one
thing moves at a time:

  1 bare_ode          diffrax, dY/dt = M @ Y          (the Step 3 baseline, complex128)
  2 lindblad_diffrax  diffrax, hand-written Lindblad, dense H and L_k
  3 dq_dense          dynamiqs mesolve, SAME H and L_k, dense layout
  4 dq_dia_cat        dynamiqs mesolve, cat qubit, sparse DIA operators, dense rho

DTYPE CHANGE: density matrices are complex128 = 16 bytes/element, so
    rho_bytes R = 16 * N^2
and every float64 number from Step 3 is re-derived, not reused.

Trajectory storage is switched off everywhere (SaveAt(t1=True) / save_states=False).
Step 3a established that peak memory is exactly flat in step count in that
configuration, which is what makes rows 3 and 4 - where the adaptive controller
chooses its own step count - comparable to the fixed-step rows 1 and 2. The step
count actually taken is recorded for every row.
"""

import argparse
import gc
import json
import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

GIB = 1024**3
MIB = 1024**2

# physical parameters
N_JUMP = 2          # number of jump operators, fixed and recorded
T_FINAL = 0.1
N_STEPS = 10
ALPHA = 2.0         # cat amplitude
K2 = 1.0            # two-photon dissipation
K1 = 0.05           # single-photon loss
DELTA = 0.0         # detuning


def stats(dev):
    return dev.memory_stats()


def show(dev, tag, base=0):
    st = stats(dev)
    print(f"    {tag:<40} in_use={st['bytes_in_use']:>15,d} "
          f"({st['bytes_in_use'] / GIB:8.5f} GiB)  "
          f"peak={st['peak_bytes_in_use']:>15,d} "
          f"({st['peak_bytes_in_use'] / GIB:8.5f} GiB)  nalloc={st['num_allocs']:>6,d}"
          + (f"  delta={st['bytes_in_use'] - base:+,d}" if base else ""), flush=True)
    return st


def tree_bytes(x) -> int:
    return sum(int(getattr(l, "nbytes", 0)) for l in jax.tree.leaves(x))


def make_operators(N: int, seed: int = 0):
    """Random dense Hermitian H and N_JUMP random dense jump operators."""
    keys = jax.random.split(jax.random.key(seed), 2 * N_JUMP + 3)
    def cplx(k1_, k2_, shape):
        return (jax.random.normal(k1_, shape, dtype=jnp.float64)
                + 1j * jax.random.normal(k2_, shape, dtype=jnp.float64))
    A = cplx(keys[0], keys[1], (N, N)) / np.sqrt(N)
    H = 0.5 * (A + A.conj().T)                       # Hermitian
    Ls = [cplx(keys[2 + 2 * k], keys[3 + 2 * k], (N, N)) / np.sqrt(N)
          for k in range(N_JUMP)]
    rho0 = jnp.zeros((N, N), dtype=jnp.complex128).at[0, 0].set(1.0 + 0j)
    for o in [H, rho0, *Ls]:
        o.block_until_ready()
    assert H.dtype == jnp.complex128 and rho0.dtype == jnp.complex128
    return H, Ls, rho0


# --------------------------------------------------------------------- row 1
def row_bare_ode(dev, N, out):
    R = 16 * N * N
    print(f"\n  --- 1 bare_ode  dY/dt = M@Y  N={N} complex128 ---", flush=True)
    print(f"    rho_bytes R = 16*{N}^2 = {R:,d} B = {R / MIB:.2f} MiB", flush=True)
    print(f"    PREDICTION (from Step 3b, Tsit5): peak = 12.393 x R = "
          f"{12.393 * R / GIB:.4f} GiB", flush=True)
    rec = {"row": "1_bare_ode", "N": N, "R": R, "pred_in_R": 12.393,
           "n_jump": 0, "status": "ok"}
    M = Y0 = sol = None
    try:
        k1_, k2_ = jax.random.split(jax.random.key(0))
        M = ((jax.random.normal(k1_, (N, N), dtype=jnp.float64)
              + 1j * jax.random.normal(k2_, (N, N), dtype=jnp.float64))
             / np.sqrt(N) - 0.5 * jnp.eye(N, dtype=jnp.complex128))
        Y0 = jnp.eye(N, dtype=jnp.complex128)
        M.block_until_ready(); Y0.block_until_ready()
        st_setup = show(dev, "after operator setup")
        base = st_setup["bytes_in_use"]
        peak_setup = st_setup["peak_bytes_in_use"]
        op_bytes = tree_bytes(M)
        print(f"    operator storage = {op_bytes:,d} B = {op_bytes / R:.4f} R",
              flush=True)

        def vf(t, y, args):
            return args @ y

        t_0 = time.perf_counter()
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vf), diffrax.Tsit5(), t0=0.0, t1=T_FINAL,
            dt0=T_FINAL / N_STEPS, y0=Y0, args=M,
            saveat=diffrax.SaveAt(t1=True), max_steps=N_STEPS + 1,
            stepsize_controller=diffrax.ConstantStepSize())
        jax.block_until_ready(sol)
        el = time.perf_counter() - t_0
        st = show(dev, "after solve", base)
        print(f"    peak / R = {st['peak_bytes_in_use'] / R:.3f}   "
              f"steps={int(sol.stats['num_steps'])}   elapsed {el:.1f}s", flush=True)
        print(f"    peak during SETUP = {peak_setup:,d} ({peak_setup / R:.3f} R); "
              f"setup-dominated = {st['peak_bytes_in_use'] == peak_setup}", flush=True)
        rec.update({"op_bytes": op_bytes, "peak": st["peak_bytes_in_use"],
                    "in_use": st["bytes_in_use"], "peak_setup": peak_setup,
                    "setup_dominated": st["peak_bytes_in_use"] == peak_setup,
                    "peak_in_R": st["peak_bytes_in_use"] / R,
                    "steps": int(sol.stats["num_steps"]), "elapsed_s": round(el, 2)})
    except Exception as exc:  # noqa: BLE001
        print(f"    !! {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        rec.update({"status": "OOM" if "RESOURCE_EXHAUSTED" in str(exc) else "ERR",
                    "exc": type(exc).__name__,
                    "peak": stats(dev)["peak_bytes_in_use"]})
    finally:
        M = Y0 = sol = None
        gc.collect(); jax.clear_caches()
        rec["clean"] = stats(dev)["bytes_in_use"] == 0
    out.append(rec)


# --------------------------------------------------------------------- row 2
def row_lindblad_diffrax(dev, N, out):
    R = 16 * N * N
    n_ops = 1 + 3 * N_JUMP     # H, and per k: L, Ldag, LdagL
    print(f"\n  --- 2 lindblad_diffrax  hand-written, dense, n_jump={N_JUMP} "
          f"N={N} ---", flush=True)
    print(f"    rho_bytes R = {R:,d} B = {R / MIB:.2f} MiB", flush=True)
    print(f"    HAND-DERIVED PREDICTION:", flush=True)
    print(f"      operators resident: H + {N_JUMP}x(L, Ldag, LdagL) = {n_ops} R",
          flush=True)
    print(f"      vector-field live temporaries: acc + (L@rho | @Ldag) + LdagL@rho"
          f" + rho@LdagL + 1 working ~ 5 R", flush=True)
    print(f"      solver (Tsit5, Step 3b): 5.393 R fixed + 7 stages x 1 R = 12.393 R,"
          f" of which 1 R was the single operator", flush=True)
    pred = 12.393 - 1 + n_ops + 5
    print(f"      TOTAL PREDICTED PEAK ~ 12.393 - 1 + {n_ops} + 5 = {pred:.3f} R = "
          f"{pred * R / GIB:.4f} GiB", flush=True)

    rec = {"row": "2_lindblad_diffrax", "N": N, "R": R, "n_jump": N_JUMP,
           "n_ops": n_ops, "pred_in_R": pred, "status": "ok"}
    H = Ls = rho0 = sol = None
    try:
        H, Ls, rho0 = make_operators(N)
        Lds = [L.conj().T for L in Ls]
        LdLs = [Ld @ L for L, Ld in zip(Ls, Lds)]
        for o in Lds + LdLs:
            o.block_until_ready()
        op_bytes = tree_bytes(H) + tree_bytes(Ls) + tree_bytes(Lds) + tree_bytes(LdLs)
        st_setup = show(dev, "after operator setup")
        base = st_setup["bytes_in_use"]
        peak_setup = st_setup["peak_bytes_in_use"]
        print(f"    operator storage = {op_bytes:,d} B = {op_bytes / R:.4f} R "
              f"(predicted {n_ops} R)", flush=True)

        def vf(t, rho, args):
            H_, Ls_, Lds_, LdLs_ = args
            drho = -1j * (H_ @ rho - rho @ H_)
            for L_, Ld_, LdL_ in zip(Ls_, Lds_, LdLs_):
                drho = drho + L_ @ rho @ Ld_ - 0.5 * (LdL_ @ rho + rho @ LdL_)
            return drho

        t_0 = time.perf_counter()
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vf), diffrax.Tsit5(), t0=0.0, t1=T_FINAL,
            dt0=T_FINAL / N_STEPS, y0=rho0, args=(H, Ls, Lds, LdLs),
            saveat=diffrax.SaveAt(t1=True), max_steps=N_STEPS + 1,
            stepsize_controller=diffrax.ConstantStepSize())
        jax.block_until_ready(sol)
        el = time.perf_counter() - t_0
        st = show(dev, "after solve", base)
        pk = st["peak_bytes_in_use"]
        print(f"    peak / R = {pk / R:.3f}  (predicted {pred:.3f})  "
              f"measured-predicted = {pk / R - pred:+.3f} R", flush=True)
        print(f"    steps={int(sol.stats['num_steps'])}  elapsed {el:.1f}s", flush=True)
        print(f"    peak during SETUP = {peak_setup:,d} ({peak_setup / R:.3f} R); "
              f"setup-dominated = {pk == peak_setup}", flush=True)
        rec.update({"op_bytes": op_bytes, "peak": pk, "in_use": st["bytes_in_use"],
                    "peak_setup": peak_setup, "setup_dominated": pk == peak_setup,
                    "peak_in_R": pk / R, "steps": int(sol.stats["num_steps"]),
                    "elapsed_s": round(el, 2)})
    except Exception as exc:  # noqa: BLE001
        print(f"    !! {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        rec.update({"status": "OOM" if "RESOURCE_EXHAUSTED" in str(exc) else "ERR",
                    "exc": type(exc).__name__,
                    "peak": stats(dev)["peak_bytes_in_use"]})
    finally:
        H = Ls = rho0 = sol = None
        gc.collect(); jax.clear_caches()
        rec["clean"] = stats(dev)["bytes_in_use"] == 0
    out.append(rec)


# --------------------------------------------------------------------- row 3/4
def row_dynamiqs(dev, N, out, mode: str):
    """mode='dense'    -> same random dense H/L as row 2
       mode='dia_cat'  -> cat qubit, operators left in their native DIA layout
       mode='cat_dense'-> the SAME cat qubit, operators forced dense via .asdense()

    cat_dense is the control for dia_cat: identical physics, identical N, identical
    solver, differing ONLY in operator layout. It is what makes the DIA saving
    attributable to the layout rather than to the change of physical model.
    """
    import dynamiqs as dq

    R = 16 * N * N
    tag = {"dense": "3_dq_dense", "dia_cat": "4_dq_dia_cat",
           "cat_dense": "5_dq_cat_DENSE_control"}[mode]
    print(f"\n  --- {tag}  N={N} ---", flush=True)
    rec = {"row": tag, "N": N, "R": R, "status": "ok", "mode": mode}
    H = jump_ops = rho0 = res = None
    try:
        if mode == "dense":
            Hm, Ls, rho0m = make_operators(N)
            H = dq.asqarray(Hm).asdense()
            jump_ops = [dq.asqarray(L).asdense() for L in Ls]
            rho0 = dq.asqarray(rho0m).asdense()
            n_jump = N_JUMP
            print(f"    same random dense H and {n_jump} jump operators as row 2",
                  flush=True)
            print(f"    PREDICTION: identical physics to row 2, so any difference "
                  f"is dynamiqs' own overhead", flush=True)
            del Hm, Ls, rho0m
        else:
            force_dense = (mode == "cat_dense")
            a = dq.destroy(N)
            H = DELTA * dq.dag(a) @ a
            L2 = np.sqrt(K2) * (a @ a - ALPHA**2 * dq.eye(N))
            L1 = np.sqrt(K1) * a
            if force_dense:
                H, L2, L1 = H.asdense(), L2.asdense(), L1.asdense()
            jump_ops = [L2, L1]
            n_jump = 2
            rho0 = dq.fock_dm(N, 0).asdense()
            nd = [getattr(o, "ndiags", None) for o in [H, L2, L1]]
            dia_pred = sum(16 * (d or 0) * N for d in nd)
            print(f"    cat qubit: H=delta a^dag a, L2=sqrt(k2)(a^2-alpha^2), "
                  f"L1=sqrt(k1) a", flush=True)
            print(f"    layouts: H={H.layout} L2={L2.layout} L1={L1.layout}  "
                  f"ndiags={nd}", flush=True)
            print(f"    PREDICTION from DIA structure: 16 * ndiags * N summed = "
                  f"{dia_pred:,d} B = {dia_pred / MIB:.4f} MiB", flush=True)
            print(f"    the same operators DENSE would cost "
                  f"{3 * R:,d} B = {3 * R / MIB:.1f} MiB  "
                  f"(saving factor {3 * R / max(dia_pred, 1):.0f}x)", flush=True)
            rec["dia_pred_bytes"] = dia_pred if not force_dense else 3 * R
            rec["ndiags"] = nd
            rec["force_dense"] = force_dense

        op_bytes = tree_bytes(H) + tree_bytes(jump_ops)
        rec["n_jump"] = n_jump
        st_setup = show(dev, "after operator setup")
        base = st_setup["bytes_in_use"]
        peak_setup = st_setup["peak_bytes_in_use"]
        print(f"    MEASURED operator storage = {op_bytes:,d} B = "
              f"{op_bytes / MIB:.4f} MiB = {op_bytes / R:.5f} R", flush=True)
        if mode in ("dia_cat", "cat_dense"):
            ratio = op_bytes / rec["dia_pred_bytes"]
            print(f"    measured / DIA prediction = {ratio:.4f}   "
                  f"-> {'DIA saving DELIVERED' if ratio < 2 else 'DENSIFIED - BUG'}",
                  flush=True)

        tsave = jnp.linspace(0.0, T_FINAL, 2)
        t_0 = time.perf_counter()
        res = dq.mesolve(H, jump_ops, rho0, tsave,
                         method=dq.method.Tsit5(rtol=1e-6, atol=1e-6,
                                                max_steps=100000),
                         save_states=False, progress_meter=False)
        jax.block_until_ready(res)
        el = time.perf_counter() - t_0
        st = show(dev, "after mesolve", base)
        pk = st["peak_bytes_in_use"]
        print(f"    peak / R = {pk / R:.3f}   elapsed {el:.1f}s", flush=True)
        print(f"    peak during SETUP = {peak_setup:,d} ({peak_setup / R:.3f} R); "
              f"setup-dominated = {pk == peak_setup}", flush=True)
        rec.update({"op_bytes": op_bytes, "peak": pk, "in_use": st["bytes_in_use"],
                    "peak_setup": peak_setup, "setup_dominated": pk == peak_setup,
                    "peak_in_R": pk / R, "elapsed_s": round(el, 2)})
    except Exception as exc:  # noqa: BLE001
        print(f"    !! {type(exc).__name__}: {str(exc)[:400]}", flush=True)
        rec.update({"status": "OOM" if "RESOURCE_EXHAUSTED" in str(exc) else "ERR",
                    "exc": type(exc).__name__,
                    "peak": stats(dev)["peak_bytes_in_use"]})
    finally:
        H = jump_ops = rho0 = res = None
        gc.collect(); jax.clear_caches()
        rec["clean"] = stats(dev)["bytes_in_use"] == 0
    out.append(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True,
                    choices=["bare", "lindblad", "dq_dense", "dq_dia",
                             "dq_cat_dense"])
    ap.add_argument("--N", type=int, default=4000)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    dev = jax.local_devices()[0]
    st = stats(dev)
    print("#" * 110, flush=True)
    print(f"STEP 4 [{args.row}]  jax {jax.__version__} diffrax {diffrax.__version__}",
          flush=True)
    try:
        import dynamiqs as dq
        print(f"  dynamiqs {dq.__version__}", flush=True)
    except Exception:  # noqa: BLE001
        pass
    for v in ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE"):
        print(f"  {v} = {os.environ.get(v, '<unset>')}", flush=True)
    print(f"  bytes_limit = {st['bytes_limit']:,d} = {st['bytes_limit'] / GIB:.4f} GiB",
          flush=True)
    print(f"  complex128 = 16 bytes/element;  R = rho_bytes = 16*N^2 = "
          f"{16 * args.N * args.N:,d} B", flush=True)
    print("#" * 110, flush=True)

    out = []
    if args.row == "bare":
        row_bare_ode(dev, args.N, out)
    elif args.row == "lindblad":
        row_lindblad_diffrax(dev, args.N, out)
    elif args.row == "dq_dense":
        row_dynamiqs(dev, args.N, out, "dense")
    elif args.row == "dq_cat_dense":
        row_dynamiqs(dev, args.N, out, "cat_dense")
    else:
        row_dynamiqs(dev, args.N, out, "dia_cat")

    with open(args.json, "a") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"\n  appended {len(out)} record(s) to {args.json}", flush=True)


if __name__ == "__main__":
    main()
