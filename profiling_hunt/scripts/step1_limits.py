"""Step 1 - Grounding: what memory does this GPU actually give us?

Reads `memory_stats()` under one env configuration (passed as argv[1] label) and
dumps every metric, plus the physical card size from pynvml/nvidia-smi, so the
ceiling used by every later step is measured rather than assumed.

Run one process per configuration: XLA env vars are read at backend init, so
they cannot be varied inside a single process.
"""

import json
import os
import subprocess
import sys

LABEL = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"
OUT = sys.argv[2] if len(sys.argv) > 2 else None

MIB = 1024**2
GIB = 1024**3

ENV_KEYS = (
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_FLAGS",
    "CUDA_VISIBLE_DEVICES",
    "SLURM_JOB_ID",
    "SLURM_JOB_GPUS",
)

METRIC_KEYS = (
    "bytes_in_use",
    "peak_bytes_in_use",
    "bytes_limit",
    "pool_bytes",
    "peak_pool_bytes",
    "largest_alloc_size",
    "num_allocs",
    "bytes_reservable_limit",
    "bytes_reserved",
    "peak_bytes_reserved",
)


def banner(title: str) -> None:
    print("=" * 78, flush=True)
    print(title, flush=True)
    print("=" * 78, flush=True)


def show_stats(stats: dict, title: str) -> None:
    print(f"\n{title}", flush=True)
    if stats is None:
        print("  <memory_stats() returned None - allocator does not report>", flush=True)
        return
    for k in sorted(stats):
        v = stats[k]
        if isinstance(v, int) and "alloc" not in k or "bytes" in k:
            print(f"  {k:26s} {v:>18,d}   {v / GIB:10.4f} GiB   {v / MIB:12.1f} MiB",
                  flush=True)
        else:
            print(f"  {k:26s} {v:>18,d}", flush=True)


def nvidia_physical() -> dict:
    """Physical card size, straight from the driver."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        print("\nnvidia-smi (index, name, total MiB, used MiB):", flush=True)
        rows = []
        for line in out.splitlines():
            print(f"  {line}", flush=True)
            parts = [p.strip() for p in line.split(",")]
            rows.append({"index": int(parts[0]), "name": parts[1],
                         "total_mib": int(parts[2]), "used_mib": int(parts[3])})
        return {"rows": rows}
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        print(f"  nvidia-smi failed: {exc!r}", flush=True)
        return {"rows": []}


def main() -> None:
    banner(f"STEP 1 - CONFIGURATION: {LABEL}")

    print("environment:", flush=True)
    for k in ENV_KEYS:
        print(f"  {k:34s} = {os.environ.get(k, '<unset>')}", flush=True)

    nv = nvidia_physical()

    # Import jax only after env is reported, so the reported env is the one used.
    import jax

    jax.config.update("jax_enable_x64", True)
    print(f"\njax {jax.__version__}   x64={jax.config.jax_enable_x64}", flush=True)
    print(f"backend: {jax.default_backend()}", flush=True)
    print(f"local_devices: {jax.local_devices()}", flush=True)

    dev = jax.local_devices()[0]
    print(f"device[0]: {dev}  kind={dev.device_kind}", flush=True)

    # (1) stats at rest, before this process has allocated anything itself
    st_rest = dev.memory_stats()
    show_stats(st_rest, "memory_stats() AT REST (no user allocation yet)")

    # (2) after one tiny allocation - this is what forces lazy init / first pool grab
    import jax.numpy as jnp

    x = jnp.ones((1024, 1024), dtype=jnp.float64)  # 8 MiB exactly
    x.block_until_ready()
    print(f"\nallocated probe array: shape={x.shape} dtype={x.dtype} "
          f"nbytes={x.nbytes:,d} ({x.nbytes / MIB:.1f} MiB)", flush=True)
    st_probe = dev.memory_stats()
    show_stats(st_probe, "memory_stats() AFTER 8 MiB float64 probe allocation")

    del x
    import gc

    gc.collect()
    jax.clear_caches()
    st_after = dev.memory_stats()
    show_stats(st_after, "memory_stats() AFTER del + clear_caches (baseline check)")

    # (3) the derived ceiling
    total_mib = nv["rows"][0]["total_mib"] if nv["rows"] else None
    limit = st_probe.get("bytes_limit") if st_probe else None

    print("\n" + "-" * 78, flush=True)
    print("DERIVED CEILING", flush=True)
    print("-" * 78, flush=True)
    if total_mib is not None:
        phys = total_mib * MIB
        print(f"  physical total        {phys:>18,d} B  {phys / GIB:10.4f} GiB  "
              f"({total_mib} MiB)", flush=True)
        for frac in (0.95, 0.90, 0.75):
            v = int(phys * frac)
            print(f"  physical * {frac:<5}      {v:>18,d} B  {v / GIB:10.4f} GiB",
                  flush=True)
    if limit is not None:
        print(f"  MEASURED bytes_limit  {limit:>18,d} B  {limit / GIB:10.4f} GiB",
              flush=True)
        if total_mib is not None:
            print(f"  measured / physical   {limit / (total_mib * MIB):.6f}", flush=True)
            print(f"  BRIEF's LIMIT_BYTES   {45_890_509_210:>18,d} B  "
                  f"{45_890_509_210 / GIB:10.4f} GiB", flush=True)
            print(f"  measured / brief      {limit / 45_890_509_210:.6f}", flush=True)

    # (4) optional: what is the largest SINGLE allocation that actually succeeds?
    # bytes_limit is what XLA *believes* it may allocate. This bisection measures
    # what the allocator actually hands over - one buffer, no transients, no kron.
    ceiling = None
    if os.environ.get("PROBE_CEILING") == "1":
        print("\n" + "-" * 78, flush=True)
        print("SINGLE-ALLOCATION CEILING PROBE (bisection on jnp.zeros float64)",
              flush=True)
        print("-" * 78, flush=True)
        import gc as _gc

        def try_alloc(nelem: int) -> tuple[bool, str]:
            buf = None
            try:
                buf = jnp.zeros(nelem, dtype=jnp.float64)
                buf.block_until_ready()
                got = dev.memory_stats()["bytes_in_use"]
                ok = got >= nelem * 8
                return ok, f"bytes_in_use={got:,d}"
            except Exception as exc:  # noqa: BLE001 - OOM is the expected outcome
                return False, type(exc).__name__
            finally:
                del buf
                _gc.collect()
                jax.clear_caches()

        limit_elems = limit // 8
        lo, hi = 0, int(limit_elems * 1.10)
        # tolerance 1 MiB = 131072 float64 elements
        tol = 131072
        n_probe = 0
        while hi - lo > tol:
            mid = (lo + hi) // 2
            ok, info = try_alloc(mid)
            n_probe += 1
            print(f"  probe {n_probe:2d}: {mid * 8:>18,d} B "
                  f"({mid * 8 / GIB:8.4f} GiB) -> {'OK  ' if ok else 'FAIL'}  {info}",
                  flush=True)
            if ok:
                lo = mid
            else:
                hi = mid
        ceiling = lo * 8
        print(f"\n  largest single allocation that SUCCEEDS: {ceiling:,d} B  "
              f"{ceiling / GIB:.4f} GiB", flush=True)
        print(f"  bytes_limit                             : {limit:,d} B  "
              f"{limit / GIB:.4f} GiB", flush=True)
        print(f"  usable / bytes_limit                    : {ceiling / limit:.6f}",
              flush=True)
        print(f"  shortfall vs bytes_limit                : {limit - ceiling:,d} B  "
              f"{(limit - ceiling) / MIB:.1f} MiB", flush=True)

    if OUT:
        rec = {
            "label": LABEL,
            "single_alloc_ceiling": ceiling,
            "env": {k: os.environ.get(k) for k in ENV_KEYS},
            "jax": jax.__version__,
            "device_kind": dev.device_kind,
            "nvidia": nv,
            "stats_at_rest": st_rest,
            "stats_after_probe": st_probe,
            "stats_after_cleanup": st_after,
        }
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"\nappended JSON record to {OUT}", flush=True)

    print(f"\nSTEP 1 CONFIG {LABEL}: DONE", flush=True)


if __name__ == "__main__":
    main()
