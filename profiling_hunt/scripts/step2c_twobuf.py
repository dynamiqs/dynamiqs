"""Step 2c - the two-buffer ceiling.

Step 1 showed a SINGLE allocation reaches 99.9997 % of bytes_limit. The Step 2b
sweep showed that the kron pattern - one resident buffer R plus a new buffer S -
dies somewhere between a combined 37.539 GiB (survived) and 40.000 GiB (failed),
i.e. 89-95 % of bytes_limit. This script measures that boundary directly.

For each resident size R, bisect the largest second buffer S that can be
allocated while R is still live, and report R + S.

If R + S is constant, the ceiling is a fixed workspace/overhead budget.
If R + S shrinks as R grows, it is fragmentation.
"""

import gc
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

GIB = 1024**3
MIB = 1024**2


def main() -> None:
    dev = jax.local_devices()[0]
    st = dev.memory_stats()
    limit = st["bytes_limit"]
    print("#" * 100, flush=True)
    print("STEP 2c - TWO-BUFFER CEILING", flush=True)
    for v in ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE"):
        print(f"  {v} = {os.environ.get(v, '<unset>')}", flush=True)
    print(f"  bytes_limit = {limit:,d} B = {limit / GIB:.4f} GiB", flush=True)
    print(f"  pool_bytes  = {st['pool_bytes']:,d} B = {st['pool_bytes'] / GIB:.4f} GiB",
          flush=True)
    print("#" * 100, flush=True)

    # resident sizes to test, in GiB - chosen to span the kron ladder's range
    resident_gib = [0.0, 0.5, 1.0, 2.0, 4.0, 7.508, 8.0, 10.125, 16.0, 20.0, 30.031]

    print(f"\n{'R GiB':>10} {'max S GiB':>12} {'R+S GiB':>12} {'(R+S)/limit':>12} "
          f"{'shortfall GiB':>14} {'probes':>7}", flush=True)

    results = []
    for rg in resident_gib:
        r_elems = int(rg * GIB / 8)
        resident = None
        try:
            if r_elems > 0:
                resident = jnp.zeros(r_elems, dtype=jnp.float64)
                resident.block_until_ready()
            r_actual = dev.memory_stats()["bytes_in_use"]

            def try_second(nelem: int) -> bool:
                buf = None
                try:
                    buf = jnp.zeros(nelem, dtype=jnp.float64)
                    buf.block_until_ready()
                    return True
                except Exception:  # noqa: BLE001 - OOM expected
                    return False
                finally:
                    del buf
                    gc.collect()

            lo, hi = 0, int((limit - r_actual) * 1.05 / 8)
            tol = 131072  # 1 MiB
            probes = 0
            while hi - lo > tol:
                mid = (lo + hi) // 2
                if try_second(mid):
                    lo = mid
                else:
                    hi = mid
                probes += 1
            s_bytes = lo * 8
            total = r_actual + s_bytes
            results.append((r_actual, s_bytes, total))
            print(f"{r_actual / GIB:>10.4f} {s_bytes / GIB:>12.4f} {total / GIB:>12.4f} "
                  f"{total / limit:>12.6f} {(limit - total) / GIB:>14.4f} {probes:>7}",
                  flush=True)

        except Exception as exc:  # noqa: BLE001
            print(f"{rg:>10.4f}  FAILED: {type(exc).__name__}: {str(exc)[:200]}",
                  flush=True)
        finally:
            del resident
            gc.collect()
            jax.clear_caches()

    if results:
        totals = [t for _, _, t in results]
        print(f"\n  R+S ranges from {min(totals) / GIB:.4f} to {max(totals) / GIB:.4f} GiB",
              flush=True)
        print(f"  spread = {(max(totals) - min(totals)) / MIB:.1f} MiB", flush=True)
        print(f"  mean (R+S)/bytes_limit = "
              f"{sum(totals) / len(totals) / limit:.6f}", flush=True)
        print(f"  => usable two-buffer budget is "
              f"{100 * sum(totals) / len(totals) / limit:.2f} % of bytes_limit",
              flush=True)


if __name__ == "__main__":
    main()
