"""Step 4c - does dynamiqs inherit the per-call retention leak of Finding 3.D?

Finding 3.D: a diffrax vector field that CAPTURES its operator leaks one full copy
of that operator per diffeqsolve call, unbounded, until jax.clear_caches().
dynamiqs builds its Lindblad vector field around H and the jump operators, so the
question is whether repeated mesolve calls with fresh operators accumulate.

Repeat mesolve N_REPEAT times with FRESH operators each time and watch bytes_in_use.
Flat  => bounded (one cache slot). Growing => the same unbounded leak.
"""

import gc
import os

import jax

jax.config.update("jax_enable_x64", True)

import dynamiqs as dq  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

GIB = 1024**3
MIB = 1024**2
N = 2000
R = 16 * N * N
N_REPEAT = 6
T_FINAL = 0.1


def one_mesolve(dev, i: int, layout: str, clear: bool):
    """One full mesolve with FRESH operators."""
    if layout == "dense":
        keys = jax.random.split(jax.random.key(3000 + i), 6)
        def cplx(a, b):
            return (jax.random.normal(a, (N, N), dtype=jnp.float64)
                    + 1j * jax.random.normal(b, (N, N), dtype=jnp.float64)) / np.sqrt(N)
        A = cplx(keys[0], keys[1])
        H = dq.asqarray(0.5 * (A + A.conj().T)).asdense()
        jump_ops = [dq.asqarray(cplx(keys[2], keys[3])).asdense(),
                    dq.asqarray(cplx(keys[4], keys[5])).asdense()]
        del A
    else:  # dia cat, with a slightly different alpha each call so ops are fresh
        a = dq.destroy(N)
        alpha = 2.0 + 0.01 * i
        H = (0.0 + 0.001 * i) * dq.dag(a) @ a
        jump_ops = [np.sqrt(1.0) * (a @ a - alpha**2 * dq.eye(N)),
                    np.sqrt(0.05) * a]

    rho0 = dq.fock_dm(N, 0).asdense()
    res = dq.mesolve(H, jump_ops, rho0, jnp.linspace(0.0, T_FINAL, 2),
                     method=dq.method.Tsit5(rtol=1e-6, atol=1e-6),
                     save_states=False, progress_meter=False)
    jax.block_until_ready(res)

    res = H = jump_ops = rho0 = None
    gc.collect()
    if clear:
        jax.clear_caches()
    st = dev.memory_stats()
    return st["bytes_in_use"], st["num_allocs"]


def main() -> None:
    dev = jax.local_devices()[0]
    print("#" * 100, flush=True)
    print("STEP 4c - does dynamiqs mesolve retain operators per call?", flush=True)
    print(f"  dynamiqs {dq.__version__}   N={N}  R = 16*N^2 = {R:,d} B "
          f"({R / MIB:.2f} MiB)", flush=True)
    for v in ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE"):
        print(f"  {v} = {os.environ.get(v, '<unset>')}", flush=True)
    print(f"  bytes_limit = {dev.memory_stats()['bytes_limit']:,d}", flush=True)
    print("#" * 100, flush=True)

    for layout in ("dense", "dia"):
        for clear in (False, True):
            print(f"\n--- layout={layout}  jax.clear_caches()={clear} ---", flush=True)
            series = []
            for i in range(N_REPEAT):
                try:
                    inuse, nalloc = one_mesolve(dev, i, layout, clear)
                    series.append(inuse)
                    print(f"    mesolve {i}: in_use={inuse:>15,d} "
                          f"({inuse / MIB:9.2f} MiB = {inuse / R:6.3f} R)  "
                          f"num_allocs={nalloc:>7,d}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"    mesolve {i}: !! {type(exc).__name__}: "
                          f"{str(exc)[:250]}", flush=True)
                    break
            if len(series) >= 3:
                tail = series[1:]
                per = (tail[-1] - tail[0]) / (len(tail) - 1)
                print(f"    tail slope = {per:+,.0f} B/call = {per / R:+.3f} R/call",
                      flush=True)
                verdict = ("BOUNDED (flat)" if abs(per) < R * 0.1
                           else "GROWING <-- LEAK")
                print(f"    VERDICT: {verdict}", flush=True)
            gc.collect()
            jax.clear_caches()
            print(f"    after config cleanup: "
                  f"in_use={dev.memory_stats()['bytes_in_use']:,d}", flush=True)


if __name__ == "__main__":
    main()
