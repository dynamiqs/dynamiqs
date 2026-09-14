"""Step 2d - is the early OOM caused by kron, or by the allocation PATTERN?

Step 2c showed two simultaneously-live buffers reach 99.9994 % of bytes_limit when
allocated into a fresh pool. Yet the kron ladder OOMed at n=32,k=11 asking for
32 GiB while holding only 8 GiB - a combination Step 2c proved allocatable
(R=8 GiB allowed S up to 34.18 GiB).

The difference can only be the HISTORY of the pool. This script replicates the
ladder's allocation pattern with plain jnp.zeros - no kron, no arithmetic - in
three variants:

  grow_hold : b = zeros(next)      # new allocated while old still bound, old
                                   # freed on rebind. Same as the kron ladder.
  grow_free : del b; b = zeros(..) # old freed BEFORE the new one is requested.
  fresh     : one single allocation of the final size into a clean pool.

If grow_hold OOMs where kron OOMed, the cause is the allocation pattern and not
kron. If grow_free then survives, the cost is specifically "hold the old buffer
while allocating the new one", which is exactly what every ODE solver stage does.
"""

import gc
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

GIB = 1024**3


def ladder(dev, n: int, kmax: int, mode: str) -> dict:
    """Run the size ladder with plain allocations. Returns summary."""
    print(f"\n--- n={n} mode={mode} ---", flush=True)
    b = None
    last_ok, oom_k = None, None
    try:
        b = jnp.zeros(n * n, dtype=jnp.float64)
        b.block_until_ready()
        for k in range(1, kmax + 1):
            nelem = n * n * 4**k
            want = nelem * 8
            try:
                if mode == "grow_free":
                    del b
                    gc.collect()
                    b = None
                nb = jnp.zeros(nelem, dtype=jnp.float64)
                nb.block_until_ready()
                b = nb
                del nb
                st = dev.memory_stats()
                print(f"  k={k:>2} want={want / GIB:8.4f} GiB  OK   "
                      f"in_use={st['bytes_in_use'] / GIB:8.4f} "
                      f"peak={st['peak_bytes_in_use'] / GIB:8.4f}", flush=True)
                last_ok = k
            except Exception as exc:  # noqa: BLE001 - OOM expected
                st = dev.memory_stats()
                print(f"  k={k:>2} want={want / GIB:8.4f} GiB  OOM  "
                      f"in_use={st['bytes_in_use'] / GIB:8.4f} "
                      f"peak={st['peak_bytes_in_use'] / GIB:8.4f}", flush=True)
                print(f"     !! {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                oom_k = k
                break
    except Exception as exc:  # noqa: BLE001
        print(f"  setup failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
        del b
        gc.collect()
        jax.clear_caches()
    print(f"  => n={n} mode={mode}: last_ok_k={last_ok} oom_k={oom_k}", flush=True)
    return {"n": n, "mode": mode, "last_ok": last_ok, "oom_k": oom_k}


def main() -> None:
    dev = jax.local_devices()[0]
    st = dev.memory_stats()
    print("#" * 100, flush=True)
    print("STEP 2d - ALLOCATION PATTERN vs kron", flush=True)
    for v in ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE"):
        print(f"  {v} = {os.environ.get(v, '<unset>')}", flush=True)
    print(f"  bytes_limit = {st['bytes_limit']:,d} B = {st['bytes_limit'] / GIB:.4f} GiB",
          flush=True)
    print("#" * 100, flush=True)

    # kron OOMed at these (n, k). Reference from job 69373.
    cases = [(32, 11), (2, 15), (16, 12), (31, 11), (9, 13)]
    out = []
    for n, kref in cases:
        print("\n" + "=" * 100, flush=True)
        print(f"CASE n={n}: kron OOMed at k={kref} "
              f"(output {8 * n * n * 4**kref / GIB:.4f} GiB, "
              f"input {8 * n * n * 4 ** (kref - 1) / GIB:.4f} GiB)", flush=True)
        print("=" * 100, flush=True)
        out.append(ladder(dev, n, kref, "grow_hold"))
        out.append(ladder(dev, n, kref, "grow_free"))

    print("\n" + "#" * 100, flush=True)
    print("SUMMARY  (compare last_ok_k against the kron reference)", flush=True)
    print("#" * 100, flush=True)
    ref = {32: 10, 2: 14, 16: 11, 31: 11, 9: 12}
    print(f"{'n':>4} {'mode':>12} {'last_ok_k':>10} {'kron last_ok_k':>16} {'verdict':>28}",
          flush=True)
    for r in out:
        kr = ref[r["n"]]
        if r["last_ok"] == kr:
            v = "same as kron"
        elif r["last_ok"] is not None and r["last_ok"] > kr:
            v = f"survives {r['last_ok'] - kr} further"
        else:
            v = "worse than kron"
        print(f"{r['n']:>4} {r['mode']:>12} {str(r['last_ok']):>10} {kr:>16} {v:>28}",
              flush=True)


if __name__ == "__main__":
    main()
