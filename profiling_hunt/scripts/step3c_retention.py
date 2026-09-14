"""Step 3c - is the post-solve retention bounded, or an unbounded leak?

Every diffeqsolve in Step 3 left exactly 256,000,000 B = M + Y0 resident after the
caller dropped every Python reference and called gc.collect() + jax.clear_caches().

Two very different explanations:
  (a) BOUNDED  - one cache slot holds the last call's arguments. Resident memory
                 stays flat across repeated solves. Annoying, not a leak.
  (b) LEAK     - every solve retains its own copy. Resident memory grows linearly
                 with the number of solves and eventually OOMs.

This repeats the same solve N_REPEAT times, building a FRESH M and Y0 each time so
that a per-call retention would accumulate, and reports bytes_in_use after each.
"""

import gc
import os

import jax

jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

GIB = 1024**3
MIB = 1024**2
N = 4000
N_REPEAT = 6
STATE = 8 * N * N


def one_solve(dev, i: int, vf_mode: str, clear: bool) -> int:
    kM, kY = jax.random.split(jax.random.key(1000 + i))  # fresh arrays every time
    M = jax.random.normal(kM, (N, N), dtype=jnp.float64) / (N**0.5)
    Y0 = jax.random.normal(kY, (N, N), dtype=jnp.float64)
    M.block_until_ready()
    Y0.block_until_ready()

    if vf_mode == "closure":
        def vf(t, y, args):
            return M @ y
        a = None
    else:
        def vf(t, y, args):
            return args @ y
        a = M

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf), diffrax.Euler(), t0=0.0, t1=1.0, dt0=0.1, y0=Y0,
        args=a, saveat=diffrax.SaveAt(t1=True), max_steps=11,
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    jax.block_until_ready(sol)

    sol = M = Y0 = a = vf = None
    gc.collect()
    if clear:
        jax.clear_caches()
    st = dev.memory_stats()
    return st["bytes_in_use"], st["num_allocs"]


def control_plain_jit(dev, i: int, vf_mode: str) -> tuple[int, int]:
    """CONTROL: no diffrax at all. A plain jax.jit function, fresh closure each call.

    If this leaks the same way, the behaviour is generic JAX jit-cache retention and
    diffrax is not responsible.
    """
    kM, kY = jax.random.split(jax.random.key(2000 + i))
    M = jax.random.normal(kM, (N, N), dtype=jnp.float64) / (N**0.5)
    Y0 = jax.random.normal(kY, (N, N), dtype=jnp.float64)
    M.block_until_ready()
    Y0.block_until_ready()

    if vf_mode == "closure":
        @jax.jit
        def f(y):
            return M @ y
        r = f(Y0)
    else:
        @jax.jit
        def f(m, y):
            return m @ y
        r = f(M, Y0)
    r.block_until_ready()

    r = M = Y0 = f = None
    gc.collect()
    st = dev.memory_stats()
    return st["bytes_in_use"], st["num_allocs"]


def main() -> None:
    dev = jax.local_devices()[0]
    print("#" * 100, flush=True)
    print("STEP 3c - POST-SOLVE RETENTION: bounded cache slot, or unbounded leak?",
          flush=True)
    print(f"  N={N}  state_bytes={STATE:,d} ({STATE / MIB:.2f} MiB)  "
          f"M+Y0 = {2 * STATE:,d} B", flush=True)
    print(f"  bytes_limit = {dev.memory_stats()['bytes_limit']:,d}", flush=True)
    print("#" * 100, flush=True)

    for vf_mode in ("closure", "args"):
        for clear in (True, False):
            print(f"\n--- vf_mode={vf_mode}  jax.clear_caches()={clear} ---",
                  flush=True)
            base = dev.memory_stats()["bytes_in_use"]
            print(f"    before any solve: in_use={base:,d}", flush=True)
            series = []
            for i in range(N_REPEAT):
                try:
                    inuse, nalloc = one_solve(dev, i, vf_mode, clear)
                    series.append(inuse)
                    print(f"    solve {i}: in_use={inuse:>15,d} "
                          f"({inuse / MIB:9.2f} MiB = {inuse / STATE:5.2f} states)  "
                          f"num_allocs={nalloc:>7,d}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"    solve {i}: !! {type(exc).__name__}: "
                          f"{str(exc)[:200]}", flush=True)
                    break
            if len(series) >= 2:
                growth = series[-1] - series[0]
                per = growth / (len(series) - 1)
                print(f"    growth across {len(series)} solves = {growth:+,d} B "
                      f"({per:+,.0f} B/solve = {per / STATE:+.3f} states/solve)",
                      flush=True)
                print(f"    VERDICT: "
                      f"{'BOUNDED (flat)' if abs(per) < STATE * 0.1 else 'GROWING <-- LEAK'}",
                      flush=True)
            # reset between configurations
            gc.collect()
            jax.clear_caches()
            print(f"    after config cleanup: "
                  f"in_use={dev.memory_stats()['bytes_in_use']:,d}", flush=True)

    # ---------------- CONTROL: plain jax.jit, no diffrax anywhere ----------------
    for vf_mode in ("closure", "args"):
        print(f"\n--- CONTROL plain jax.jit  vf_mode={vf_mode}  "
              f"(no clear_caches) ---", flush=True)
        series = []
        for i in range(N_REPEAT):
            try:
                inuse, nalloc = control_plain_jit(dev, i, vf_mode)
                series.append(inuse)
                print(f"    call {i}: in_use={inuse:>15,d} "
                      f"({inuse / MIB:9.2f} MiB = {inuse / STATE:5.2f} states)  "
                      f"num_allocs={nalloc:>7,d}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"    call {i}: !! {type(exc).__name__}: {str(exc)[:200]}",
                      flush=True)
                break
        if len(series) >= 3:
            # use the tail: the first call includes one-off warm-up
            tail = series[1:]
            per = (tail[-1] - tail[0]) / (len(tail) - 1)
            print(f"    tail growth = {per:+,.0f} B/call = "
                  f"{per / STATE:+.3f} states/call", flush=True)
            print(f"    VERDICT: "
                  f"{'BOUNDED (flat)' if abs(per) < STATE * 0.1 else 'GROWING <-- LEAK'}",
                  flush=True)
        gc.collect()
        jax.clear_caches()
        print(f"    after config cleanup: "
              f"in_use={dev.memory_stats()['bytes_in_use']:,d}", flush=True)


if __name__ == "__main__":
    main()
