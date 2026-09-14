"""Step 2 - Validating the instrument: is memory_stats() a faithful OOM proxy?

Geometric Kronecker growth with a FIXED small right operand:

    A_step = jax.random.normal(key,  (2, 2))    # never grows
    b      = jax.random.normal(key2, (n, n))    # seed, swept
    for k in range(K_MAX):
        b = jnp.kron(b, A_step)                 # side x2, elements x4

    predicted_bytes(n, k) = 8 * n^2 * 4^k       # float64, 8 bytes/element

Every iteration logs every metric from the brief's vocabulary plus the
measured/predicted ratios, to stdout and to a CSV.

Note on jit: jnp.kron carries its own internal @jax.jit decorator (it is defined
that way in jax.numpy). No jit is added here, and because memory_stats() is
queried after every individual kron call, per-operation allocation visibility is
preserved - which is what the no-jit rule is protecting.

Usage:  step2_kron.py --n 37 --kmax 16 --csv out.csv [--repeat 1] [--kstop K]
        step2_kron.py --n-range 2 40 --kmax 16 --csv out.csv
"""

import argparse
import csv
import gc
import os
import sys
import time
import traceback

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  - must follow the x64 config update

try:
    from jaxlib.xla_extension import XlaRuntimeError
except Exception:  # noqa: BLE001 - fall back if the symbol moves
    XlaRuntimeError = RuntimeError

GIB = 1024**3
MIB = 1024**2

METRIC_KEYS = (
    "bytes_in_use",
    "peak_bytes_in_use",
    "bytes_limit",
    "pool_bytes",
    "peak_pool_bytes",
    "largest_alloc_size",
    "num_allocs",
    "largest_free_block_bytes",
)

CSV_FIELDS = (
    "n", "k", "side", "elements", "predicted_bytes", "status", "exc_type",
    *METRIC_KEYS,
    "ratio_inuse", "ratio_peak", "elapsed_s", "repeat",
)


def predicted_bytes(n: int, k: int) -> int:
    """float64 -> 8 bytes per element."""
    return 8 * n * n * 4**k


def is_oom(exc: BaseException) -> bool:
    s = str(exc)
    return ("RESOURCE_EXHAUSTED" in s or "out of memory" in s.lower()
            or "Out of memory" in s)


def fmt_row(rec: dict) -> str:
    p = rec["predicted_bytes"]
    return (
        f"n={rec['n']:>3} k={rec['k']:>3} side={rec['side']:>7} "
        f"elem={rec['elements']:>15,d} "
        f"pred={p:>15,d} ({p / GIB:8.4f} GiB) "
        f"| {rec['status']:<4} "
        f"| in_use={rec['bytes_in_use']:>15,d} ({rec['bytes_in_use'] / GIB:8.4f} GiB) "
        f"peak={rec['peak_bytes_in_use']:>15,d} ({rec['peak_bytes_in_use'] / GIB:8.4f} GiB) "
        f"limit={rec['bytes_limit']:>15,d} "
        f"pool={rec['pool_bytes']:>15,d} peak_pool={rec['peak_pool_bytes']:>15,d} "
        f"largest={rec['largest_alloc_size']:>15,d} "
        f"free_blk={rec['largest_free_block_bytes']:>15,d} "
        f"nalloc={rec['num_allocs']:>6} "
        f"| in_use/pred={rec['ratio_inuse']:>7} peak/pred={rec['ratio_peak']:>7} "
        f"| {rec['elapsed_s']:.3f}s"
    )


def run_one_n(dev, n: int, kmax: int, writer, fh, repeat: int = 0,
              kstop: int | None = None) -> dict:
    """Run the kron ladder for one seed size n. Returns a summary dict."""
    print("\n" + "=" * 120, flush=True)
    print(f"SEED n={n}  (repeat {repeat})   kmax={kmax}"
          + (f"   kstop={kstop}" if kstop is not None else ""), flush=True)
    print("=" * 120, flush=True)

    base = dev.memory_stats()
    print(f"  baseline before ladder: bytes_in_use={base['bytes_in_use']:,d} "
          f"num_allocs={base['num_allocs']:,d} "
          f"peak_bytes_in_use={base['peak_bytes_in_use']:,d}", flush=True)

    summary = {"n": n, "repeat": repeat, "oom_k": None, "last_ok_k": None,
               "last_ok_bytes": None, "exc_type": None}

    A_step = None
    b = None
    try:
        A_step = jax.random.normal(jax.random.key(1), (2, 2), dtype=jnp.float64)
        A_step.block_until_ready()
        b = jax.random.normal(jax.random.key(0), (n, n), dtype=jnp.float64)
        b.block_until_ready()
        assert b.dtype == jnp.float64, f"dtype is {b.dtype}, expected float64"
        assert A_step.dtype == jnp.float64

        # k = 0 row: the seed itself, before any kron
        st = dev.memory_stats()
        rec = {
            "n": n, "k": 0, "side": n, "elements": n * n,
            "predicted_bytes": predicted_bytes(n, 0), "status": "seed",
            "exc_type": "", **{m: st.get(m, -1) for m in METRIC_KEYS},
            "ratio_inuse": round(st["bytes_in_use"] / predicted_bytes(n, 0), 4),
            "ratio_peak": round(st["peak_bytes_in_use"] / predicted_bytes(n, 0), 4),
            "elapsed_s": 0.0, "repeat": repeat,
        }
        print("  " + fmt_row(rec), flush=True)
        writer.writerow(rec)
        fh.flush()

        upper = kmax if kstop is None else min(kmax, kstop)
        for k in range(1, upper + 1):
            side = n * 2**k
            elems = n * n * 4**k
            pred = predicted_bytes(n, k)
            t0 = time.perf_counter()
            try:
                b = jnp.kron(b, A_step)
                b.block_until_ready()
                dt = time.perf_counter() - t0
                st = dev.memory_stats()
                rec = {
                    "n": n, "k": k, "side": side, "elements": elems,
                    "predicted_bytes": pred, "status": "ok", "exc_type": "",
                    **{m: st.get(m, -1) for m in METRIC_KEYS},
                    "ratio_inuse": round(st["bytes_in_use"] / pred, 4),
                    "ratio_peak": round(st["peak_bytes_in_use"] / pred, 4),
                    "elapsed_s": round(dt, 4), "repeat": repeat,
                }
                print("  " + fmt_row(rec), flush=True)
                writer.writerow(rec)
                fh.flush()
                summary["last_ok_k"] = k
                summary["last_ok_bytes"] = st["bytes_in_use"]

            except (XlaRuntimeError, Exception) as exc:  # noqa: B014,BLE001
                dt = time.perf_counter() - t0
                st = dev.memory_stats()
                etype = type(exc).__name__
                oom = is_oom(exc)
                rec = {
                    "n": n, "k": k, "side": side, "elements": elems,
                    "predicted_bytes": pred,
                    "status": "OOM" if oom else "ERR", "exc_type": etype,
                    **{m: st.get(m, -1) for m in METRIC_KEYS},
                    "ratio_inuse": round(st["bytes_in_use"] / pred, 4),
                    "ratio_peak": round(st["peak_bytes_in_use"] / pred, 4),
                    "elapsed_s": round(dt, 4), "repeat": repeat,
                }
                print("  " + fmt_row(rec), flush=True)
                print(f"    !! {etype}: {str(exc)[:400]}", flush=True)
                writer.writerow(rec)
                fh.flush()
                summary["oom_k"] = k
                summary["exc_type"] = etype
                break

    except Exception as exc:  # noqa: BLE001 - never let one n kill the sweep
        print(f"  !! unexpected failure outside the ladder for n={n}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        summary["exc_type"] = type(exc).__name__

    finally:
        # Clean between configurations and confirm the return to baseline.
        del b, A_step
        gc.collect()
        jax.clear_caches()
        after = dev.memory_stats()
        print(f"  after cleanup: bytes_in_use={after['bytes_in_use']:,d} "
              f"num_allocs={after['num_allocs']:,d} "
              f"peak_bytes_in_use={after['peak_bytes_in_use']:,d} "
              f"pool_bytes={after['pool_bytes']:,d}", flush=True)
        clean = after["bytes_in_use"] == 0
        print(f"  returned to baseline: {'YES' if clean else 'NO  <-- DIRTY'}"
              f"  (delta vs pre-ladder = "
              f"{after['bytes_in_use'] - base['bytes_in_use']:+,d} B)", flush=True)
        summary["clean"] = clean
        summary["allocs_delta"] = after["num_allocs"] - base["num_allocs"]

    print(f"  SUMMARY n={n}: last_ok_k={summary['last_ok_k']} "
          f"oom_k={summary['oom_k']} exc={summary['exc_type']} "
          f"clean={summary.get('clean')} allocs_delta={summary.get('allocs_delta')}",
          flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--n-range", type=int, nargs=2, default=None)
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--kstop", type=int, default=None,
                    help="stop at this k even if it would fit (for targeted probes)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--csv", type=str, required=True)
    args = ap.parse_args()

    if args.n is not None:
        ns = [args.n]
    elif args.n_range is not None:
        ns = list(range(args.n_range[0], args.n_range[1] + 1))
    else:
        ap.error("give --n or --n-range")

    dev = jax.local_devices()[0]
    print("#" * 120, flush=True)
    print(f"STEP 2 - kron ladder.  jax {jax.__version__}  x64={jax.config.jax_enable_x64}",
          flush=True)
    print(f"device {dev} kind={dev.device_kind}", flush=True)
    for v in ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE"):
        print(f"  {v} = {os.environ.get(v, '<unset>')}", flush=True)
    st0 = dev.memory_stats()
    print(f"  bytes_limit = {st0['bytes_limit']:,d} B = {st0['bytes_limit'] / GIB:.4f} GiB",
          flush=True)
    print(f"  pool_bytes  = {st0['pool_bytes']:,d} B = {st0['pool_bytes'] / GIB:.4f} GiB",
          flush=True)
    print(f"  predicted_bytes(n,k) = 8 * n^2 * 4^k   [float64 = 8 bytes/element]",
          flush=True)
    print("#" * 120, flush=True)

    new_file = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        summaries = []
        for r in range(args.repeat):
            for n in ns:
                summaries.append(
                    run_one_n(dev, n, args.kmax, writer, fh, repeat=r,
                              kstop=args.kstop))

    print("\n" + "#" * 120, flush=True)
    print("SWEEP SUMMARY", flush=True)
    print("#" * 120, flush=True)
    for s in summaries:
        print(f"  n={s['n']:>3} rep={s['repeat']} last_ok_k={s['last_ok_k']} "
              f"oom_k={s['oom_k']} exc={s['exc_type']} clean={s.get('clean')}",
              flush=True)


if __name__ == "__main__":
    main()
